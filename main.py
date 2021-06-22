from DRIT import DRIT
import argparse
from utils import *

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of DRIT"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test, guide]')
    parser.add_argument('--dataset', type=str, default='VIF', help='dataset_name')
    parser.add_argument('--task', type=str, default='VIF', help='fusion_task')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='using learning rate decay')

    parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')
    parser.add_argument('--weight_epoch', type=int, default=3, help='The number of epochs to optimize weight_block')
    parser.add_argument('--decay_epoch', type=int, default=0, help='The number of decay epochs to run')
    # parser.add_argument('--iteration', type=int, default=500, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=50, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=40, help='The number of ckpt_save_freq')

    parser.add_argument('--num_attribute', type=int, default=1, help='number of attributes to sample')
    parser.add_argument('--direction', type=str, default='a2b', help='direction of guided image translation')
    parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')

    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--content_adv_w', type=int, default=1, help='weight of content adversarial loss')
    parser.add_argument('--domain_adv_w', type=int, default=0, help='weight of domain adversarial loss')
    parser.add_argument('--recon_w', type=int, default=10, help='weight of self-reconstruction loss')
    parser.add_argument('--chro_w', type=int, default=0, help='weight of chrominance loss')
    parser.add_argument('--per_w', type=int, default=0, help='weight of perceptual loss')
    parser.add_argument('--kl_w', type=float, default=0.001, help='weight of kl-divergence loss')

    parser.add_argument('--ch', type=int, default=48, help='base channel number per layer')
    parser.add_argument('--concat', type=str2bool, default = True, help='using concat networks')


    parser.add_argument('--n_z', type=int, default=96, help='length of z')
    parser.add_argument('--n_layer', type=int, default=4, help='number of layers in G, D')

    parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')

    # If you don't use multi-discriminator, then recommend n_dis = 6

    parser.add_argument('--n_scale', type=int, default=1, help='number of scales for discriminator')

    # using the multiscale discriminator often gets better results

    parser.add_argument('--n_d_con', type=int, default=2, help='# of iterations for updating content discriminator')

    # model can still generate diverse results with n_d_con = 1

    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral normalization')

    parser.add_argument('--img_size', type=int, default=192, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=1, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = DRIT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        # show_all_variables()

        if args.phase == 'train':
            # launch the graph in a session
            gan.train()
            print(" [*] Training DRF finished!")
            # gan.train_weight_block()
            # print(" [*] Training weight_block finished!")
        #
        if args.phase == 'test':
            gan.test()
            print(" [*] Test finished!")


if __name__ == '__main__':
    main()
