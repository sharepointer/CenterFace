import os
import shutil
import argparse

args = argparse.ArgumentParser()
args.add_argument('--score', default=0.3, type=float)
args.add_argument('--show', default=0, type=int)
ARGS = args.parse_args()

detect_score = ARGS.score
is_show = ARGS.show

my_path = os.getcwd()

print('detect....')
dst_dir = '/home/work/Blue/FaceDetection/CenterFace/FDDB'
src_dir = os.path.join(my_path, '../simulation')

# angle_list = [0, 90, 180, 270]
angle_list = [0]
for angle in angle_list:
    os.system('python model_test.py --score={} --angle={} --show={}'.format(detect_score, angle, is_show))
    name = 'fddb_output_{}.txt'.format(angle)
    shutil.copy(os.path.join(src_dir, name), os.path.join(dst_dir, name))

print('evaluate...')
os.chdir(dst_dir)
cmd = 'python3 val.py'
os.system(cmd)

print('plot roc...')
cmd = 'gnuplot plot.p'
os.system(cmd)
