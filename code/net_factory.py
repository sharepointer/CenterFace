import net_centerface as centerface


def build(inputs, is_training=False, ssh_head=True, num_head_ch=24, upsample_bilinear=False, backbone='mbv2_1.0'):
    return centerface.build(inputs, is_training, ssh_head, num_head_ch, upsample_bilinear, backbone)
