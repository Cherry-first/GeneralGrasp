import os
import numpy as np
import math
import cv2
import cliport.utils.visual_utils as vu
import pdb

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.tasks import cameras
from cliport.utils import utils
from cliport.models.core.attention import Attention
from cliport.models.core.transport import Transport
from cliport.models.streams.two_stream_attention import TwoStreamAttention
from cliport.models.streams.two_stream_transport import TwoStreamTransport

from cliport.models.streams.two_stream_attention import TwoStreamAttentionLat
from cliport.models.streams.two_stream_transport import TwoStreamTransportLat
from cliport.models.streams.conceptfusion import conceptfusion


class TransporterAgent(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']
        self.save_visuals = 0
        self._build_model()
        self.automatic_optimization = False
        #self._set_optimizers()
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

        self.total_steps = 0
        self.warmup_epochs = self.cfg['train']['warmup_epochs']
        self.sch = cfg['train']['lr_scheduler']
        self.lr = cfg['train']['lr']
        self.lr_min = cfg['train']['lr_min']

    def _build_model(self):
        self.attention = None
        self.transport = None
        
        raise NotImplementedError()

    def configure_optimizers(self):
        
        opt_attn = torch.optim.AdamW(self.attention.parameters(), lr=self.cfg['train']['lr'], betas=(0.9, 0.95))
        opt_trans = torch.optim.AdamW(self.transport.parameters(), lr=self.cfg['train']['lr'], betas=(0.9, 0.95)) 
        self.max_epochs = self.trainer.max_epochs
        if self.sch:
            print('Using cosine annealing learning rate scheduler with warm up !')
        
            def sch_foo(epoch):
                """Decay the learning rate with half-cycle cosine after warmup"""
                if epoch < self.warmup_epochs:
                    lr = self.lr * (epoch+1) / self.warmup_epochs 
                else:
                    lr = self.lr_min + (self.lr - self.lr_min) * 0.5 * \
                        (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
                return lr/self.lr
            
            lrs_attn = torch.optim.lr_scheduler.LambdaLR(opt_attn, lr_lambda=sch_foo)
            lrs_trans = torch.optim.lr_scheduler.LambdaLR(opt_trans, lr_lambda=sch_foo)
            return (
                {"optimizer": opt_attn, 
                "lr_scheduler": lrs_attn},
                {"optimizer": opt_trans, 
                "lr_scheduler": lrs_trans}
            )

        else :
            return [opt_attn, opt_trans]
         
    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        # pdb.set_trace()
        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, compute_err=False, return_output=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        loss, err = self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)
        if not return_output:
            return loss, err
        else:
            return loss, err, out

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta, return_output=False):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        # Backpropagate.
        if backprop:
            attn_optim, _ = self.optimizers()
            if self.sch:
                s_att, _ = self.lr_schedulers()
                s_att.step(epoch=self.current_epoch)
            self.manual_backward(loss)
            attn_optim.step()
            attn_optim.zero_grad()
        # pdb.set_trace()
        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
            dis_ord = 1 if not return_output else 2
            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=dis_ord),
                'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss, err

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def transport_training_step(self, frame, backprop=True, compute_err=False, return_output=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        inp = {'inp_img': inp_img, 'p0': p0}
        output = self.trans_forward(inp, softmax=False)
        
        if not return_output: # during training
            err, loss = self.transport_criterion(backprop, compute_err, inp, output, p0, p1, p1_theta)
            return loss, err
        else: # for real image visualization only
            err, loss, idx = self.transport_criterion(backprop, compute_err, inp, output, p0, p1, p1_theta, return_output)
            return loss, err, output[0,idx,:,:]
        
    def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta, return_output=False):
        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))
        loss = self.cross_entropy_with_logits(output, label)
        if backprop:
            _, transport_optim = self.optimizers()
            if self.sch:
                _, s_trans = self.lr_schedulers()
                s_trans.step(epoch=self.current_epoch)
            # pdb.set_trace()
            self.manual_backward(loss)
            transport_optim.step()
            transport_optim.zero_grad()
 
        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf = self.trans_forward(inp)
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
            dis_ord = 1 if not return_output else 2
            err = {
                'dist': np.linalg.norm(np.array(q) - p1_pix, ord=dis_ord),
                'theta': np.absolute((theta - p1_theta) % np.pi)
            }
        self.transport.iters += 1
        if not return_output:
            return err, loss
        else:
            return err, loss, argmax[2]


    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()
        
        frame, _ = batch  # keys of frame: ['img', 'p0', 'p0_theta', 'p1', 'p1_theta', 'perturb_params', 'lang_goal']

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
    
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = int( self.trainer.global_step / 2)

        # self.trainer.train_loop.running_loss.append(total_loss)
        # pdb.set_trace()
        # self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        # pdb.set_trace()
        global_step = self.total_steps 
        if global_step in self.save_steps:
            print('Save in check_save_iteration, the total_step is {}'.format(global_step))
            train_loss = self.trainer.callback_metrics['tr/loss']
            steps = f'{global_step :05d}'
            filename = f"steps={steps}-train_loss={train_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if global_step % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.save_visuals = 0
        self.val_output_list = []

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.val_output_list.append(
            dict(
                val_loss=val_total_loss,
                val_loss0=loss0,
                val_loss1=loss1,
                val_attn_dist_err=err0['dist'],
                val_attn_theta_err=err0['theta'],
                val_trans_dist_err=err1['dist'],
                val_trans_theta_err=err1['theta'],
            )
        )
    
    # def on_validation_batch_end(self):
    #     val_output = self.val_output_list[-1]
    #     val_batch_loss = val_output['val_loss'].item()
    #     if self.total_steps in self.save_steps:
    #         print('Save the checkpoint after validate, the save_step is {}'.format(self.total_steps))
    #         print('The global_step: {}, The total_step: {}'.format(self.trainer.global_step, self.total_steps))
    #         # train_loss = self.trainer.callback_metrics['tr/loss']
    #         steps = f'{self.total_steps :05d}'
    #         filename = f"steps={steps}-val_loss={val_batch_loss:0.8f}.ckpt"
    #         checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
    #         ckpt_path = os.path.join(checkpoint_path, filename)
    #         self.trainer.save_checkpoint(ckpt_path)
    #     if self.total_steps % 1000 == 0:
    #         self.save_last_checkpoint()
    
    def on_train_epoch_end(self):
        utils.set_seed(self.trainer.current_epoch+1)

    def on_validation_epoch_end(self):
        all_outputs = self.val_output_list
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))
        # print('The global_step: {}, The total_step: {}'.format(self.trainer.global_step, self.total_steps))

        self.val_output_list.clear()
        print('The reminder! The current total_steps is {}'.format(self.total_steps))
        if self.total_steps in self.save_steps:
            print('Save the checkpoint after validate, the save_step is {}'.format(self.total_steps))
            print('The global_step: {}, The total_step: {}'.format(self.trainer.global_step, self.total_steps))
            # train_loss = self.trainer.callback_metrics['tr/loss']
            steps = f'{self.total_steps :05d}'
            filename = f"steps={steps}-val_loss={mean_val_total_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)
        if self.total_steps % 1000 == 0:
            self.save_last_checkpoint()

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)

        # Attention model forward pass.
        pick_inp = {'inp_img': img}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': p0_pix,
            'place': p1_pix,
        }

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
        self.to(device=self.device_type)

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.on_test=True
        self.test_output_list = []
        self.save_heatmap = []

    def test_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0, out_attn = self.attn_training_step(frame, backprop=False, compute_err=True, return_output=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1, out_attn = self.attn_training_step(frame, backprop=False, compute_err=True, return_output=True)
                loss1 += l1
            else:
                l1, err1, out_trans = self.transport_training_step(frame, backprop=False, compute_err=True, return_output=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1        

        #import pdb; pdb.set_trace()
        img = frame['img'][:,:,:3]
        pick_place = frame['p0']
        place_place = frame['p1']
        pick_radius = frame['pick_radius']
        place_radius = frame['place_radius']
        text = frame['lang_goal']

        img = img.astype(np.uint8)

        #save heatmap images
        out_attn = out_attn.reshape(320,160).detach().cpu().numpy()
        out_trans = out_trans.detach().cpu().numpy()
        save_path = os.path.join(self.cfg['train']['train_dir'], 'real_vis')
        os.makedirs(save_path, exist_ok=True)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        name = self.name.split('-')[0]
        save = vu.save_tensor_with_heatmap(image, out_attn,
            f'{save_path}/{name}_pick{batch_idx + 1:06d}.png',
            l=text)
        save = vu.save_tensor_with_heatmap(image, out_trans,
            f'{save_path}/{name}_place{batch_idx + 1:06d}.png',
            l=text)
        # save gt images
        # brg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.circle(brg, (pick_place[1], pick_place[0]), int(pick_radius), (0, 255, 0), 2)
        # cv2.circle(brg, (place_place[1], place_place[0]), int(place_radius), (0, 0, 255), 2)
        # cv2.putText(brg, text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        # foler = 'data_gt_real_images'
        # idx = len(os.listdir(foler))
        # cv2.imwrite(f'data_gt_real_images/real{idx}.png', brg)
        

        # whether successful pick and place ?
        if err0['dist'] < frame['pick_radius']:
            success_pick = 1
        else:
            success_pick = 0
        
        if err1['dist'] < frame['place_radius']:
            success_place = 1
        else:
            success_place = 0
        
        if err0['dist'] < frame['pick_radius'] and err1['dist'] < frame['place_radius']:
            success = 1
        else:
            success = 0
                
        self.test_output_list.append( 
                dict(
                val_loss=val_total_loss,
                val_loss0=loss0,
                val_loss1=loss1,
                val_attn_dist_err=err0['dist'],
                val_attn_theta_err=err0['theta'],
                val_trans_dist_err=err1['dist'],
                val_trans_theta_err=err1['theta'],
                success=success,
                success_pick=success_pick,
                success_place=success_place
            )
        )

    def on_test_epoch_end(self):
        all_outputs = self.test_output_list
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])
        success_rate = np.sum([v['success'] for v in all_outputs]) / len(all_outputs)
        success_pick_rate = np.sum([v['success_pick'] for v in all_outputs]) / len(all_outputs)
        success_place_rate = np.sum([v['success_place'] for v in all_outputs]) / len(all_outputs)

        file_name = os.path.join(self.cfg['train']['train_dir'],'checkpoints', 'pick_n_place_loss.txt') 
        saved_file = open(file_name, 'a')
        print('=============================', file=saved_file)
        print('vl/attn/loss', mean_val_loss0, file=saved_file)
        print('vl/trans/loss', mean_val_loss1,file=saved_file)
        print('vl/loss', mean_val_total_loss,file=saved_file)
        print('vl/total_attn_dist_err', total_attn_dist_err,file=saved_file)
        print('vl/total_attn_theta_err', total_attn_theta_err,file=saved_file)
        print('vl/total_trans_dist_err', total_trans_dist_err,file=saved_file)
        print('vl/total_trans_theta_err', total_trans_theta_err,file=saved_file)
        print('success_rate', success_rate,file=saved_file)
        print('success_pick_rate', success_pick_rate,file=saved_file)
        print('success_place_rate', success_place_rate,file=saved_file)
        saved_file.close()

    


class OriginalTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class ClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'clip_unet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetLatTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_unet_lat'
        self.attention = TwoStreamAttentionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipWithoutSkipsTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_woskip'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamRN50BertUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'rn50_bert_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
