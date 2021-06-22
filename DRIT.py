from ops import *
from utils import *
from glob import glob
import time
from datetime import datetime
import h5py
from scipy.misc import imread, imsave, imresize
from tqdm import tqdm
import scipy.io as scio
# from VGGnet.vgg16 import Vgg16
# from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch


class DRIT(object):
    def __init__(self, sess, args):
        self.model_name = 'DRIT'
        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.task = args.task
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.weight_epoch = args.weight_epoch
        # self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.num_attribute = args.num_attribute  # for test
        self.guide_img = args.guide_img
        self.direction = args.direction

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.content_init_lr = args.lr / 1.5
        self.ch = args.ch
        self.concat = args.concat
        self.task = args.task

        """ Weight """
        self.content_adv_w = args.content_adv_w
        self.domain_adv_w = args.domain_adv_w
        self.recon_w = args.recon_w
        self.kl_w = args.kl_w
        self.chro_w = args.chro_w
        self.per_w = args.per_w


        """ Generator """
        self.n_layer = args.n_layer
        self.n_z = args.n_z

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale
        self.n_d_con = args.n_d_con
        self.multi = True if args.n_scale > 1 else False
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/ue'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/gt'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# decay_flag : ", self.decay_flag)
        print("# epoch : ", self.epoch)
        print("# weight epoch : ", self.weight_epoch)
        print("# decay_epoch : ", self.decay_epoch)
        print("# attribute in test phase : ", self.num_attribute)

        print()

        print("##### Generator #####")
        print("# layer : ", self.n_layer)
        print("# z dimension : ", self.n_z)
        print("# concat : ", self.concat)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# multi-scale Dis : ", self.n_scale)
        print("# updating iteration of con_dis : ", self.n_d_con)
        print("# spectral_norm : ", self.sn)

        print()

        print("##### Weight #####")
        print("# domain_adv_weight : ", self.domain_adv_w)
        print("# content_adv_weight : ", self.content_adv_w)
        print("# recon_weight : ", self.recon_w)
        print("# kl_weight : ", self.kl_w)
        print("# chro_weight: ", self.chro_w)
        print("# per_weight: ", self.per_w)

    ##################################################################################
    # Encoder and Decoders
    ##################################################################################

    def content_encoder(self, x, is_training=True, reuse=False, res_reuse=False, scope='content_encoder'):
        channel=self.ch
        x_out=[]
        with tf.variable_scope(scope, reuse=reuse):
            x = resblock(x, channel, scope='resblock0')
            # x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv1')
            x = lrelu(x, 0.01)
            x_out.append(x)
            
            for i in range(2):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x,0.01)
                channel = channel * 2

            for i in range(1, self.n_layer):
                x = resblock(x, channel, scope='resblock_' + str(i))


        with tf.variable_scope('content_encoder_share', reuse = res_reuse):
            x = resblock(x, channel, scope='resblock_share', reuse = res_reuse)
            # x = gaussian_noise_layer(x, is_training)
            x_out.append(x)
            return x_out

    def attribute_encoder(self, x, reuse=False, scope='attribute_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=5, stride=2, pad=2, pad_type='reflect', scope='conv')
            x = lrelu(x,0.01)
            channel = channel * 2

            x = conv(x, channel, kernel=5, stride=2, pad=2, pad_type='reflect', scope='conv_0')
            x = lrelu(x,0.01)
            channel = channel * 2

            for i in range(1, self.n_layer):
                x = conv(x, channel, kernel=5, stride=2, pad=2, pad_type='reflect', scope='conv_' + str(i))
                x = lrelu(x,0.01)

            x = global_avg_pooling(x)
            x = conv(x, channels=self.n_z, kernel=1, stride=1, scope='attribute_logit')
            return x

    def attribute_encoder_concat(self, x, reuse=False, scope='attribute_encoder_concat'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=5, stride=2, pad=2, pad_type='reflect', scope='conv1')
            x = conv(x, channel, kernel=5, stride=2, pad=2, pad_type='reflect', scope='conv2')

            for i in range(1, 3): # self.n_layer):
                channel = channel * (i + 1)
                x = basic_block(x, channel, scope='basic_block_' + str(i))

            x = lrelu(x, 0.01)
            x = global_avg_pooling(x)

            mean = fully_conneted(x, channels=self.n_z, scope='z_mean')
            logvar = fully_conneted(x, channels=self.n_z, scope='z_logvar')

            return mean, logvar

    def MLP(self, z, reuse=False, scope='MLP'):
        channel = self.ch * self.n_layer
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2):
                z = fully_conneted(z, channel, scope='fully_' + str(i))
                z = relu(z)

            z = fully_conneted(z, channel * self.n_layer, scope='fully_logit')

            return z


    def generator(self, x, reuse=False, scope="generator"):
        channel = self.ch * self.n_layer
        content_layers=len(x)
        
        x_c = x[content_layers-1]
        x_d2 = x[0]
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2):
                if i==0:
                    x_c = deconv(x_c, channel // 2, kernel=5, stride=2, scope='deconv_' + str(i))
                else:
                    x_c = deconv(tf.concat([x_c, x_d2], axis=-1), channel // 2, kernel=5, stride=2, scope='deconv_' + str(i))
                x_c = layer_norm(x_c, scope='layer_norm_' + str(i))
                x_c = lrelu(x_c,0.01)
                channel = channel // 2

            x = conv(x_c, channels=self.img_ch*8, kernel=3, stride=1, pad=1, pad_type='reflect', scope='G_logit1')
            x = lrelu(x, 0.01)
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, pad_type='reflect', scope='G_logit2')
            x = tanh(x)
            return x



    def generator_concat(self, x, z, reuse=False, res_reuse=False, scope='generator_concat'):
        channel = self.ch * self.n_layer
        content_layers=len(x)
        x_c = x[content_layers-1]
        x_init = x[0]

        with tf.variable_scope('generator_concat_share', reuse=res_reuse):
            x_c = resblock(x_c, channel, reuse=res_reuse, scope='resblock')

        with tf.variable_scope(scope, reuse=reuse):
            channel = channel + self.n_z
            x_c = expand_concat(x_c, z)
            
            for i in range(1, self.n_layer):
                x_c = resblock(x_c, channel, scope='resblock_' + str(i))

            for i in range(2):
                channel = channel // 2
                x_c = deconv(x_c, channel, kernel=3, stride=2, scope='deconv_' + str(i))
                x_c = layer_norm(x_c, scope='layer_norm_' + str(i))
                x_c = lrelu(x_c, 0.01)


            
            x = conv(tf.concat([x_c, x_init], axis=-1), channels=self.ch*4, kernel=3, stride=1, pad=1, pad_type='reflect', scope='G_logit1')
            x = lrelu(x, 0.01)
            x = conv(x, channels=self.ch*2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='G_logit2')
            x = lrelu(x, 0.01)
            x = conv(x, channels=self.ch, kernel=3, stride=1, pad=1, pad_type='reflect', scope='G_logit3')
            x = lrelu(x, 0.01)
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, pad_type='reflect', scope='G_logit4')
            x = tanh(x)
            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def content_discriminator(self, x, reuse=False, scope='content_discriminator'):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch * self.n_layer
            for i in range(3):
                x = conv(x, channel, kernel=7, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x, 0.01)

            x = conv(x, channel, kernel=4, stride=1, scope='conv_3')
            x = lrelu(x, 0.01)

            x = conv(x, channels=1, kernel=1, stride=1, scope='D_content_logit')
            D_logit.append(x)

            return D_logit

    def multi_discriminator(self, x_init, reuse=False, scope="multi_discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            for scale in range(self.n_scale):
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                         scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.01)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                             scope='ms_' + str(scale) + 'conv_' + str(i))
                    x = lrelu(x, 0.01)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='ms_' + str(scale) + 'D_logit')
                D_logit.append(tf.reduce_mean(x))

                x_init = down_sample(x_init)
            return D_logit

    def discriminator(self, x, reuse=False, scope="discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv1')
            x = lrelu(x, 0.01)
            x = conv(x, channel*2, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv2')
            x = lrelu(x, 0.01)
            x = conv(x, channel*2, kernel=3, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv3')
            x = lrelu(x, 0.01)

            for i in range(3, self.n_dis):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                         scope='conv_' + str(i))
                x = lrelu(x, 0.01)
                channel = channel * 2

            x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='D_logit')
            D_logit.append(x)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def Encoder_A(self, x_A, is_training=True, random_fake=False, reuse=False, res_reuse = False):
        mean = None
        logvar = None

        content_A_ms = self.content_encoder(x_A, is_training=is_training, reuse=reuse, res_reuse=res_reuse, scope='content_encoder_A')
        attribute_A = self.attribute_encoder(x_A, reuse=reuse, scope='attribute_encoder_A')

        return content_A_ms, attribute_A #, mean, logvar

    def Encoder_B(self, x_B, is_training=True, random_fake=False, reuse=False, res_reuse=False):
        mean = None
        logvar = None

        content_B_ms = self.content_encoder(x_B, is_training=is_training, reuse=reuse, res_reuse=res_reuse, scope='content_encoder_B')
        attribute_B = self.attribute_encoder(x_B, reuse=reuse, scope='attribute_encoder_B')

        return content_B_ms, attribute_B#, mean, logvar

    def Decoder_A(self, content_B_ms, attribute_A, reuse=False, res_reuse=False):
        # x = fake_A, identity_A, random_fake_A
        # x = (B, A), (A, A), (B, z)
        
        if self.concat:
            x = self.generator_concat(x=content_B_ms, z=attribute_A, reuse=reuse, res_reuse=res_reuse, scope='generator_concat_A')
        else:
            x = self.generator(x=content_B_ms, z=attribute_A, reuse=reuse, scope='generator_A')

        return x

    def Decoder_B(self, content_A_ms, attribute_B, reuse=False, res_reuse=False):
        # x = fake_B, identity_B, random_fake_B
        # x = (A, B), (B, B), (A, z)
        if self.concat:
            x = self.generator_concat(x=content_A_ms, z=attribute_B, reuse=reuse, res_reuse=res_reuse, scope='generator_concat_B')
        else:
            x = self.generator(x=content_A_ms, reuse=reuse, scope='generator_B')

        return x

    def weight_block(self, attribute_a, attribute_b, scope="weight_block", reuse=False):
        x=tf.concat([attribute_a,attribute_b], axis=-1)
        with tf.variable_scope(scope):
            x = fully_conneted(x, channels=self.n_z, scope='MLP1', reuse=reuse)
            x = lrelu(x, 0.01)
            x = fully_conneted(x, channels=self.n_z/2, scope='MLP2', reuse=reuse)
            x = lrelu(x, 0.01)
            x = fully_conneted(x, channels=2, scope='MLP3', reuse=reuse)
            x = tf.reduce_mean(x, axis=[1, 2])
            x = relu(x)
            x = tf.nn.softmax(x)
        return x

    def discriminate_real(self, x_A, x_B):
        if self.multi:
            real_A_logit = self.multi_discriminator(x_A, scope='multi_discriminator_A')
            real_B_logit = self.multi_discriminator(x_B, scope='multi_discriminator_B')

        else:
            real_A_logit = self.discriminator(x_A, scope="discriminator_A")
            real_B_logit = self.discriminator(x_B, scope="discriminator_B")
        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        if self.multi:
            fake_A_logit = self.multi_discriminator(x_ba, reuse=True, scope='multi_discriminator_A')
            fake_B_logit = self.multi_discriminator(x_ab, reuse=True, scope='multi_discriminator_B')

        else:
            fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
            fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")
        return fake_A_logit, fake_B_logit

    def discriminate_content(self, content_A, content_B, reuse=False):
        content_A_logit = self.content_discriminator(content_A, reuse=reuse, scope='content_discriminator')
        content_B_logit = self.content_discriminator(content_B, reuse=True, scope='content_discriminator')

        return content_A_logit, content_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.content_lr = tf.placeholder(tf.float32, name='content_lr')

        """ Input Image"""
        # with tf.Graph().as_default(), tf.Session() as sess:
        self.domain_A = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.img_ch), name='domain_A')
        self.domain_B = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.img_ch), name='domain_B')

        # Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)
        #
        # trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        # trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)
        #
        #
        # trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(
        #     map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16,
        #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        # trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(
        #     map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16,
        #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        # # trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(
        # #     map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16,
        # #                   drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        #
        # trainA_iterator = trainA.make_one_shot_iterator()
        # trainB_iterator = trainB.make_one_shot_iterator()
        #
        # self.domain_A = trainA_iterator.get_next()
        # self.domain_B = trainB_iterator.get_next()

        print("domain_A", self.domain_A)

        """ Define Encoder, Generator, Discriminator """
        random_z = tf.random_normal(shape=[self.batch_size, self.n_z], mean=0.0, stddev=1.0, dtype=tf.float32)


        with tf.device('/gpu:0'):
            content_a_ms, attribute_a = self.Encoder_A(self.domain_A)
            # content_b, attribute_b, mean_b, logvar_b = self.Encoder_B(self.domain_B, res_reuse=True)
            content_b_ms, attribute_b = self.Encoder_B(self.domain_B, res_reuse=True)
            attribute_b = attribute_b + 0.2
            self.attribute_a = attribute_a
            self.attribute_b = attribute_b
            self.content_a_ms = content_a_ms
            self.content_b_ms = content_b_ms

        with tf.device('/gpu:1'):
            fake_a = self.Decoder_A(content_B_ms=content_b_ms, attribute_A=attribute_a, reuse=False, res_reuse=False)
            fake_b = self.Decoder_A(content_B_ms=content_a_ms, attribute_A=attribute_b, reuse=True, res_reuse=True)
            recon_a = self.Decoder_A(content_B_ms=content_a_ms, attribute_A=attribute_a, reuse=True, res_reuse=True)
            recon_b = self.Decoder_A(content_B_ms=content_b_ms, attribute_A=attribute_b, reuse=True, res_reuse=True)


        with tf.device('/gpu:1'):
            self.attribute_weights = self.weight_block(attribute_a=self.attribute_a, attribute_b=self.attribute_b)

            content_layers = len(content_a_ms)
            fused_content_ms = []
            for scale_num in range(len(content_a_ms)):
                fused_content_ms.append(0.5 * content_a_ms[scale_num] + 0.5 * content_b_ms[scale_num])
            for n in range(self.batch_size):
                if n==0:
                    fused_attribute = tf.expand_dims(self.attribute_weights[n, 0] * attribute_a[n,:,:,:] + self.attribute_weights[n, 1] * attribute_b[n,:,:,:], axis=0)
                else:
                    fused_attribute = tf.concat([fused_attribute, tf.expand_dims(self.attribute_weights[n, 0] * attribute_a[n,:,:,:] + self.attribute_weights[n, 1] * attribute_b[n,:,:,:], axis=0)], axis=0)

        with tf.device('/gpu:1'):
            self.fused_result = self.Decoder_A(content_B_ms=fused_content_ms, attribute_A=fused_attribute, reuse=True, res_reuse=True)

        Loss_MG = tf.reduce_mean(MG(self.fused_result))
        Loss_SD = tf.reduce_mean(SD(self.fused_result))
        self.weight_loss = - Loss_SD #Loss_MG


        """ Image """
        self.fake_A = fake_a
        self.fake_B = fake_b

        self.real_A = self.domain_A
        self.real_B = self.domain_B
        
        self.recon_A = recon_a
        self.recon_B = recon_b

        """ Define Loss """
        g_adv_loss_a = L1_loss(self.fake_A, self.domain_A)
        g_adv_loss_b = L1_loss(self.fake_B, self.domain_B)

        g_con_loss_a = L1_loss(content_a_ms[content_layers-1], content_b_ms[content_layers-1])
        g_con_loss_b = L1_loss(content_a_ms[content_layers-1], content_b_ms[content_layers-1])

        '''L_recon'''
        g_rec_loss_a = L1_loss(recon_a, self.domain_A)
        g_rec_loss_b = L1_loss(recon_b, self.domain_B)

        g_kl_loss_a = l2_regularize(attribute_a)
        g_kl_loss_b = l2_regularize(attribute_b)

        Generator_A_content_loss = self.content_adv_w * g_con_loss_a
        Generator_A_recon_loss = self.recon_w * g_rec_loss_a
        Generator_A_kl_loss = self.kl_w * g_kl_loss_a
        Generator_A_adv_loss = self.domain_adv_w * g_adv_loss_a

        self.Generator_A_loss = Generator_A_recon_loss + Generator_A_kl_loss + \
                                Generator_A_adv_loss + Generator_A_content_loss

        Generator_B_content_loss = self.content_adv_w * g_con_loss_b
        Generator_B_recon_loss = self.recon_w * g_rec_loss_b
        Generator_B_kl_loss = self.kl_w * g_kl_loss_b
        Generator_B_adv_loss = self.domain_adv_w * g_adv_loss_b

        self.Generator_B_loss = Generator_B_recon_loss + Generator_B_kl_loss + \
                                Generator_B_adv_loss + Generator_B_content_loss

        self.Generator_loss = self.Generator_A_loss + self.Generator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'encoder' in var.name or 'generator' in var.name]
        weight_vars = [var for var in t_vars if 'weight_block' in var.name]

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.Generator_loss, G_vars), clip_norm=10)
        grads_weight, _ = tf.clip_by_global_norm(tf.gradients(self.weight_loss, weight_vars), clip_norm=10)

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5, beta2 = 0.999).minimize(self.Generator_loss,
                                                                                            var_list = G_vars)
        self.weight_optim = tf.train.AdamOptimizer(self.lr, beta1 = 0.5, beta2 = 0.999).minimize(self.weight_loss,
                                                                                                 var_list = weight_vars)




        """" Summary """
        self.lr_write = tf.summary.scalar("learning_rate", self.lr)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
#        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.G_A_loss = tf.summary.scalar("G_A_loss", self.Generator_A_loss)
        # self.G_A_domain_loss = tf.summary.scalar("G_A_domain_loss", Generator_A_domain_loss)
        self.G_A_content_loss = tf.summary.scalar("G_A_content_loss", Generator_A_content_loss)
        # self.G_A_cycle_loss = tf.summary.scalar("G_A_cycle_loss", Generator_A_cycle_loss)
        self.G_A_recon_loss = tf.summary.scalar("G_A_recon_loss", Generator_A_recon_loss)
        # self.G_A_chro_loss = tf.summary.scalar("G_A_chro_loss", Generator_A_chro_loss)
        self.G_A_kl_loss = tf.summary.scalar("G_A_kl_loss", Generator_A_kl_loss)
        self.G_A_adv_loss = tf.summary.scalar("G_A_adv_loss", Generator_A_adv_loss)
        # self.G_A_per_loss = tf.summary.scalar("G_A_per_loss", Generator_A_per_loss)

        self.G_B_loss = tf.summary.scalar("G_B_loss", self.Generator_B_loss)

        self.G_B_content_loss = tf.summary.scalar("G_B_content_loss", Generator_B_content_loss)

        self.G_B_recon_loss = tf.summary.scalar("G_B_recon_loss", Generator_B_recon_loss)

        self.G_B_adv_loss = tf.summary.scalar("G_B_adv_loss", Generator_B_adv_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss,
                                        self.G_A_recon_loss,
                                        self.G_A_kl_loss, self.G_A_adv_loss, self.G_A_content_loss,

                                        self.G_B_loss,
                                        self.G_B_content_loss,
                                        self.G_B_recon_loss,
                                        self.G_B_adv_loss,
                                        self.all_G_loss])


    def train(self):
        if self.img_ch == 1:
            f = h5py.File(self.task + '.h5', 'r')
        if self.img_ch == 3:
            f = h5py.File(self.task + '_RGB.h5', 'r')
        sources = f['data'][:]
        sources = np.transpose(sources, (0, 3, 2, 1))
        num_imgs = sources.shape[0]
        mod = num_imgs % self.batch_size
        n_batches = int(num_imgs // self.batch_size)
        print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
        self.iteration=n_batches

        if mod > 0:
            print('Train set has been trimmed %d samples...\n' % mod)
            sources = sources[:-mod]
        print("source shape:", sources.shape)

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=3)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = datetime.now()
        lr = self.init_lr
        content_lr = self.content_init_lr
        for epoch in range(start_epoch, self.epoch):
            np.random.shuffle(sources)
            
            for idx in range(start_batch_id, self.iteration):
                if self.img_ch == 1:
                    patch_A= np.expand_dims(sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 0], axis=-1)
                    patch_B = np.expand_dims(sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 1], axis=-1)
                if self.img_ch == 3:
                    patch_A = sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 0:3]
                    patch_B = sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 3:6]
                         
                patch_A = (patch_A-0.5) * 2
                patch_B = (patch_B-0.5) * 2

                if self.decay_flag:
                    lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * 0.998 ** ((idx + (epoch-self.decay_epoch) * self.iteration)/20)
                    content_lr = self.content_init_lr if epoch < self.decay_epoch else self.content_init_lr * 0.998 ** ((idx + (epoch-self.decay_epoch) * self.iteration) / 20)
                train_feed_dict = {
                    self.lr: lr,
                    self.content_lr: content_lr,
                    self.domain_A: patch_A,
                    self.domain_B: patch_B
                }

                summary_str = self.sess.run(self.lr_write, feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                self.sess.run(self.G_optim, feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                if np.mod(idx + 1, self.print_freq) == 0:
                    g_loss, g_a_loss, g_b_loss = self.sess.run([self.Generator_loss, self.Generator_A_loss, self.Generator_B_loss], feed_dict=train_feed_dict)

                    self.writer.add_summary(summary_str, counter)

                    print("Epoch: [%2d] [%5d/%5d] " % (epoch, idx + 1, self.iteration), "time: ", datetime.now() - start_time, "learning rate:", lr)
                    print("g_loss: %2.5f, g_a_loss: %2.5f, g_b_loss: %2.5f\n" % (g_loss, g_a_loss, g_b_loss))
                else:
                    print("Epoch: [%2d] [%5d/%5d] " % (epoch, idx + 1, self.iteration), "time: ",
                          datetime.now() - start_time, "learning rate:", lr)


                if np.mod(idx + 1, self.print_freq) == 0:
                    batch_A_images, batch_B_images, fake_A, fake_B, recon_A, recon_B, summary_str = self.sess.run(
                        [self.real_A, self.real_B, self.fake_A, self.fake_B, self.recon_A, self.recon_B, self.G_loss], feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:01d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    save_images(batch_B_images, [self.batch_size, 1], './{}/real_B_{:01d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))

                    save_images(fake_A, [self.batch_size, 1], './{}/fake_A_{:01d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:01d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    
                    save_images(recon_A, [self.batch_size, 1], './{}/recon_A_{:01d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    save_images(recon_B, [self.batch_size, 1],
                                './{}/recon_B_{:01d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))


                # display training status
                counter += 1

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    def train_weight_block(self):
        if self.img_ch == 1:
            f = h5py.File(self.task + '.h5', 'r')
        if self.img_ch == 3:
            f = h5py.File(self.task + '_RGB.h5', 'r')
        sources = f['data'][:]
        sources = np.transpose(sources, (0, 3, 2, 1))
        num_imgs = sources.shape[0]
        mod = num_imgs % self.batch_size
        n_batches = int(num_imgs // self.batch_size)
        print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
        self.iteration = n_batches

        if mod > 0:
            print('Train set has been trimmed %d samples...\n' % mod)
            sources = sources[:-mod]
        print("source shape:", sources.shape)

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=3)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = datetime.now()
        lr = self.init_lr
        content_lr = self.content_init_lr
        for epoch in range(start_epoch, start_epoch + self.weight_epoch):
            np.random.shuffle(sources)
            for idx in range(start_batch_id, self.iteration):
                if self.img_ch == 1:
                    patch_A = np.expand_dims(
                        sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 0], axis=-1)
                    patch_B = np.expand_dims(
                        sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 1], axis=-1)
                if self.img_ch == 3:
                    patch_A = sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 0:3]
                    patch_B = sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 3:6]

                patch_A = (patch_A - 0.5) * 2
                patch_B = (patch_B - 0.5) * 2

                if self.decay_flag:
                    lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * 0.995 ** (
                                (idx + (epoch - start_epoch - self.decay_epoch) * self.iteration) / 20)

                    content_lr = self.content_init_lr if epoch < self.decay_epoch else self.content_init_lr * 0.995 ** (
                                (idx + (epoch - start_epoch - self.decay_epoch) * self.iteration) / 20)
                train_feed_dict = {
                    self.lr: lr,
                    self.content_lr: content_lr,
                    self.domain_A: patch_A,
                    self.domain_B: patch_B
                }

                summary_str = self.sess.run(self.lr_write, feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                self.sess.run(self.weight_optim, feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                if np.mod(idx + 1, self.print_freq) == 0:
                    wloss,w1,w2=self.sess.run([self.weight_loss, tf.reduce_mean(self.attribute_weights[:,0]), tf.reduce_mean(self.attribute_weights[:,1])], feed_dict = train_feed_dict)
                    print("Epoch: [%2d] [%5d/%5d] " % (epoch, idx + 1, self.iteration), "time: ",
                          datetime.now() - start_time, "learning rate:", lr)
                    print("weight loss: %s, w1: %s, w2: %s\n" % (wloss,w1,w2))

                if np.mod(idx + 1, self.print_freq) == 0:
                    batch_A_images, batch_B_images, fake_A, fake_B, recon_A, recon_B, summary_str = self.sess.run(
                        [self.real_A, self.real_B, self.fake_A, self.fake_B, self.recon_A, self.recon_B, self.G_loss],
                        feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)
            

    @property
    def model_dir(self):
        if self.concat:
            concat = "_concat"
        else:
            concat = ""

        if self.sn:
            sn = "_sn"
        else:
            sn = ""

        return "{}{}_{}_{}_{}layer_{}dis_{}scale_{}con{}".format(self.model_name, concat, self.task,
                                                                 self.gan_type,
                                                                 self.n_layer, self.n_dis, self.n_scale, self.n_d_con,
                                                                 sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0





    def test(self):
        tf.global_variables_initializer().run()
        if self.dataset_name == 'VIF':
            test_A_files = glob('./dataset/{}/Vis/*.*'.format(self.dataset_name))
            test_B_files = glob('./dataset/{}/IR/*.*'.format(self.dataset_name))


        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir2 = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir2)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        print("There are %d files to be processed." % len(test_A_files))

        for file_index in tqdm(range(len(test_A_files))):  # A -> B
            file_A=test_A_files[file_index]
            file_B=test_B_files[file_index]
            test_img_A, h, w = load_test_data(file_A, size=self.img_size)
            test_img_B, h, w = load_test_data(file_B, size=self.img_size)
            
            print("Processing A image: " + file_A + "  shape:", test_img_A.shape)
            print("Processing B image: " + file_B + "  shape:", test_img_B.shape)
            
            h = h//4 * 4
            w = w//4 * 4
                        
            file_name = os.path.basename(file_B).split(".")[0]
            file_extension = os.path.basename(file_B).split(".")[1]

            self.test_img_A = tf.placeholder(tf.float32, [1, h, w, self.img_ch], name='test_image_A')
            self.test_img_B = tf.placeholder(tf.float32, [1, h, w, self.img_ch], name='test_image_B')

            test_content_a_ms, test_attribute_a = self.Encoder_A(self.test_img_A, is_training=False, reuse=True, res_reuse=True)
            test_content_b_ms, test_attribute_b = self.Encoder_B(self.test_img_B, is_training=False, reuse=True, res_reuse=True)

            # test_attribute_weights = self.weight_block(attribute_a=test_attribute_a, attribute_b=test_attribute_b, reuse=True)

            # fuse attribute
            w_attr = 0.3
            combine_attribute = w_attr * test_attribute_a + (1-w_attr) * test_attribute_b


            # fuse content

            test_img_A_tensor = np.expand_dims(np.expand_dims(test_img_A[0:h, 0:w], axis = -1), axis = 0)
            test_img_B_tensor = np.expand_dims(np.expand_dims(test_img_B[0:h, 0:w], axis = -1), axis = 0)
            FEED_DICT = {self.test_img_A: test_img_A_tensor, self.test_img_B: test_img_B_tensor}

            content_a, content_b = self.sess.run([test_content_a_ms, test_content_b_ms], feed_dict = FEED_DICT)

            combine_content_0 = 0.5 * content_a[0] + 0.5 * content_b[0]  # L1_norm(content_a[0], content_b[0])#
            combine_content_1 = 0.5 * content_a[1] + 0.5 * content_b[1]  # L1_norm(content_a[1], content_b[1])#

            self.combine_content0 = tf.placeholder(tf.float32, [1, h, w, self.ch], name = 'content0')
            self.combine_content1 = tf.placeholder(tf.float32, [1, h // 4, w // 4, self.ch * self.n_layer],
                                                   name = 'content1')
            self.combine_content = []
            self.combine_content.append(self.combine_content0)
            self.combine_content.append(self.combine_content1)

            self.fusion_img = self.Decoder_A(content_B_ms = self.combine_content, attribute_A = combine_attribute,
                                             reuse = True, res_reuse = True)
            self.test_recon_A = self.Decoder_A(content_B_ms = test_content_a_ms, attribute_A = test_attribute_a,
                                               reuse = True,
                                               res_reuse = True)
            self.test_recon_B = self.Decoder_A(content_B_ms = test_content_b_ms, attribute_A = test_attribute_b,
                                               reuse = True,
                                               res_reuse = True)
            self.test_fake_A = self.Decoder_A(content_B_ms = test_content_b_ms, attribute_A = test_attribute_a,
                                              reuse = True,
                                              res_reuse = True)
            self.test_fake_B = self.Decoder_A(content_B_ms = test_content_a_ms, attribute_A = test_attribute_b,
                                              reuse = True,
                                              res_reuse = True)

            fake_img = self.sess.run(self.fusion_img,
                                     feed_dict = {self.test_img_A: test_img_A_tensor,
                                                  self.test_img_B: test_img_B_tensor,
                                                  self.combine_content0: combine_content_0,
                                                  self.combine_content1: combine_content_1})

            fake_img = fake_img[0, :, :, 0]

            image_path = os.path.join(self.result_dir, self.dataset_name, '{}.{}'.format(file_name, file_extension))

            imsave(image_path, (fake_img + 1.) / 2)


    def guide_test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        attribute_file = np.asarray(load_test_data(self.guide_img, size=self.img_size))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir, 'guide')
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        if self.direction == 'a2b':
            for sample_file in test_A_files:  # A -> B
                print('Processing A image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_B, feed_dict={self.content_image: sample_image,
                                                                       self.attribute_image: attribute_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                            '../../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                            '../../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")

        else:
            for sample_file in test_B_files:  # B -> A
                print('Processing B image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_A, feed_dict={self.content_image: sample_image,
                                                                       self.attribute_image: attribute_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                            '../../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write(
                    "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                            '../../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")
        index.close()
