# directly tranform group to png format for BSR evaluate
import numpy as np
import json
import scipy.io as scio
import svgwrite
from cairosvg import svg2png
import os
import cv2


test_data_dir = '/import/vision-datasets001/kl303/PG_data/PG_ndjson/fine_tuning1/'
svg_file_dir = '/import/vision-datasets001/kl303/PG_data/group_svg/'
out_put_file = '/import/vision-datasets/kl303/PycharmProjects/BSR/bench/pre_label/cluster/'
GT_file = '/import/vision-datasets/kl303/PycharmProjects/BSR/bench/PG_data/'
# out_put_file =GT_file+'new_groundtruth/'

def data2abspoints(data):
    abspoints = []
    stroke_group = []
    stroke_group_idx = -1
    abs_x = 25
    abs_y = 25
    next_is_gap = 1
    for line_data in data:

        offset_x = line_data[0]
        offset_y = line_data[1]
        if next_is_gap:
            begin_point = [abs_x+offset_x,abs_y+offset_y]
            stroke_group_idx+=1
        else:
            begin_point = [abs_x,abs_y]
        end_point = [abs_x+offset_x,abs_y+offset_y]
        abs_x +=offset_x
        abs_y +=offset_y
        abspoints.append([begin_point,end_point])
        stroke_group.append(stroke_group_idx)
        next_is_gap = line_data[2]


    return abspoints,stroke_group


def draw_sketch_with_strokes(data,svg_name,stroke_group,evaluate_file_path,img_file_name=None):

    assert len(data)==len(stroke_group)
    dims = (256,256)
    abspoints,stroke_group=data2abspoints(data)
    # abspoints, _ = data2abspoints(data)
    unique_stroke_group = np.unique(stroke_group)
    bsd_data = np.zeros((256,256),dtype=np.uint8)
    image_data = np.zeros((256,256),dtype=np.uint8)+255

    for group_idx in unique_stroke_group:
        stroke_svg_name = svg_name+'_'+str(group_idx)+'.svg'
        stroke_png_name = svg_name+'_'+str(group_idx)+'.png'
        dwg = svgwrite.Drawing(stroke_svg_name, size=dims)
        dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
        same_group_stroke_idxs = np.where(stroke_group==group_idx)[0]
        for idx in same_group_stroke_idxs:
            begin_point = abspoints[idx][0]
            end_point = abspoints[idx][1]
            dwg.add(dwg.line((begin_point[0], begin_point[1]), (end_point[0], end_point[1]),stroke=svgwrite.rgb(0, 0, 0, '%')))
        dwg.save()
        svg2png(url=stroke_svg_name,write_to=stroke_png_name)
        os.remove(stroke_svg_name)
        stroke_data = cv2.imread(stroke_png_name,0)
        os.remove(stroke_png_name)
        # fg_idx = np.where(stroke_data<25)[0]
        bsd_data[stroke_data<=240] = group_idx+1
        image_data[stroke_data<=240] = 0
    # cv2.imwrite(img_file_name,image_data)
    scio.savemat(evaluate_file_path,{'label_matrix':bsd_data})


test_datasets = ['angel','bulldozer','drill','flower','house']
# test_datasets = ['airplane','alarm-clock','ambulance','ant','apple','backpack','basket','butterfly','cactus',
#                  'campfire','candle','coffee-cup','crab','duck','face','ice-cream','pig','pineapple','suitcase','calculator','angel','bulldozer','drill','flower','house']
train_strokes = None
valid_strokes = None
eval_strokes = None
testsss_strokes =None


for dataset in test_datasets:
    with open(test_data_dir + dataset + '.ndjson', 'r') as f:
        ori_data = json.load(f)
        train_stroke = ori_data['train_data'][:650]
        valid_stroke = ori_data['train_data'][650:700]
        eval_stroke = ori_data['train_data'][700:]
        category = dataset



        for idx,stroke in enumerate(eval_stroke):

            if os.path.exists(out_put_file + category) == False:
                os.mkdir(out_put_file + category)
                os.mkdir(GT_file+'new_image/'+category)
            test_file_name = '/import/vision-datasets/kl303/PycharmProjects/BSR/bench/PG_data/test_file/' + category + '.txt'
            test_f = open(test_file_name, 'r')
            lines = test_f.readlines()
            line = lines[np.mod(idx, 100)].strip()
            mat_file_name = out_put_file + category + '/' + line[:-4] + '.mat'
            img_file_name = GT_file+'new_image/'+category+ '/' + line[:-4] + '.png'
            test_f.close()

            svg_name = svg_file_dir+str(idx)
            stroke_group = np.asarray(stroke)[:,3]
            draw_sketch_with_strokes(stroke,svg_name,stroke_group.astype(int),mat_file_name,img_file_name)
    # if train_strokes is None:
    #     train_strokes = train_stroke
    # else:
    #     train_strokes = np.concatenate((train_strokes, train_stroke))
    # if valid_strokes is None:
    #     valid_strokes = valid_stroke
    # else:
    #     valid_strokes = np.concatenate((valid_strokes, valid_stroke))
    # if eval_strokes is None:
    #     eval_strokes = eval_stroke
    # else:
    #     eval_strokes = np.concatenate((eval_strokes, eval_stroke))



