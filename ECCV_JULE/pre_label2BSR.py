# directly tranform group to png format for BSR evaluate
import numpy as np
import json
import scipy.io as scio
import svgwrite
from cairosvg import svg2png
import os
import cv2
import pdb


def data2abspoints(data):
    abspoints = []
    abs_x = 25
    abs_y = 25
    next_is_gap = 1
    for line_data in data:

        offset_x = line_data[0]
        offset_y = line_data[1]
        # if next_is_gap:
        #     begin_point = [abs_x+offset_x,abs_y+offset_y]
        # else:
        #     begin_point = [abs_x,abs_y]
        begin_point = [abs_x, abs_y]
        end_point = [abs_x+offset_x,abs_y+offset_y]
        abs_x +=offset_x
        abs_y +=offset_y
        if next_is_gap==0:
            abspoints.append([begin_point,end_point])
	else:
            abspoints.append([begin_point,begin_point])
        next_is_gap = line_data[2]


    return abspoints


def draw_sketch_with_strokes(data,stroke_group,evaluate_file_path):

    # assert len(data)==len(stroke_group)
    dims = (256,256)
    abspoints=data2abspoints(data)
    unique_stroke_group = np.unique(stroke_group)
    bsd_data = np.zeros((256,256),dtype=np.uint8)
    # image_data = np.zeros((256,256),dtype=np.uint8)+255

    for group_idx in unique_stroke_group:
        stroke_svg_name = str(group_idx)+'.svg'
        stroke_png_name = str(group_idx)+'.png'
        dwg = svgwrite.Drawing(stroke_svg_name, size=dims)
        dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
        same_group_stroke_idxs = np.where(stroke_group==group_idx)[0]
       # pdb.set_trace()
        for idx in same_group_stroke_idxs:
            if max(same_group_stroke_idxs)>=len(abspoints):
                pdb.set_trace()
            begin_point = abspoints[idx][0]
            end_point = abspoints[idx][1]
            dwg.add(dwg.line((begin_point[0], begin_point[1]), (end_point[0], end_point[1]),stroke=svgwrite.rgb(0, 0, 0, '%')))
        dwg.save()
        svg2png(url=stroke_svg_name,write_to=stroke_png_name)
        os.remove(stroke_svg_name)
        stroke_data = cv2.imread(stroke_png_name,0)
        os.remove(stroke_png_name)
        # fg_idx = np.where(stroke_data<25)[0]
        bsd_data[stroke_data<245] = group_idx+1
        # image_data[stroke_data<245] = 0
    # cv2.imwrite(img_file_name,image_data)
    scio.savemat(evaluate_file_path,{'label_matrix':bsd_data})
