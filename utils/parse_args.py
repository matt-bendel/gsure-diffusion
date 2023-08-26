import pathlib

from utils.args import Args


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    # GAN ARGS
    # parser.add_argument('--num-iters-discriminator', type=int, default=1,
    #                     help='Number of iterations of the discriminator')

    # LEARNING ARGS
    # parser.add_argument('--batch-size', default=20, type=int, help='Mini batch size')
    # parser.add_argument('--im-size', default=128, type=int, help='Mini batch size')
    # parser.add_argument('--calib_width', default=10, type=int, help='Mini batch size')
    # parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    # parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    # parser.add_argument('--beta_1', type=float, default=0, help='Beta 1 for Adam')
    # parser.add_argument('--beta_2', type=float, default=0.99, help='Beta 2 for Adam')
    # parser.add_argument('--adv-weight', type=float, default=1e-3, help='Weight for adversarial loss')
    # parser.add_argument('--var-weight', type=float, default=0.01, help='Weight for variance reward')
    # parser.add_argument('--ssim-weight', type=float, default=0.84, help='Weight for supervised loss')
    # parser.add_argument('--gp-weight', type=float, default=10, help='Weight for Gradient Penalty')

    # TODO: UPDATE DESCRIPTIONS AND MOVE GAN ARGS TO CFG FILES
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--resume-epoch', default=0, type=int, help='Mini batch size')
    parser.add_argument('--mri', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--inpaint', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--cs', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--sr', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--rcgan', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--awgn', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--dp', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--default-model-descriptor', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--exp-name', type=str, default="", help='Weight for Gradient Penalty', required=True)
    parser.add_argument('--num-noise', default=0, type=int, help='Mini batch size') # 0 Vanilla, 1 Measured, 2 Nonmeasured
    parser.add_argument('--noise-structure', default=0, type=int, help='Mini batch size')
    parser.add_argument('--num-gpus', default=1, type=int, help='Mini batch size')
    parser.add_argument('--R', default=8, type=int, help='Mini batch size')
    parser.add_argument('--mask-type', default=2, type=int, help='Mini batch size')
    parser.add_argument('--sr-scale', default=4, type=int, help='Mini batch size')

    return parser
