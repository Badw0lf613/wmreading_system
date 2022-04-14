from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
import pandas as pd
import numpy as np

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_subdirs_without_labels(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            # st.write('bd', bd)
            result.append(bd)
    return result

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs_without_labels(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':

    st.title('WMRsystem based on YOLOv5')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/best.pt', help='model.pt path(s)')
                        # default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    source = ("图片检测", "视频检测")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('开始检测'):

            detect(opt)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    img_tmp = ''
                    # st.write('get_detection_folder', os.listdir(get_detection_folder()))
                    st.header('下为检测后的图片')
                    for img in os.listdir(get_detection_folder()):
                        # txtpath = str(Path(f'{get_detection_folder()}').split('/')
                        if img != 'labels':
                            img_tmp = img
                            # st.write('img_tmp', img_tmp)
                            st.image(str(Path(f'{get_detection_folder()}') / img))
                    # 对图片路径做处理得到txt路径
                    txtpath = str(Path(f'{get_detection_folder()}') / img)
                    txtpath_list = txtpath.split('/')[0:-1]
                    # st.write('txtpath_list',txtpath_list)
                    txtpath = ''
                    for l in txtpath_list:
                        txtpath = txtpath + l + '/'
                    # st.write('img_tmp',img_tmp)
                    txtpath = txtpath + 'labels/' + img_tmp
                    txtpath = txtpath.replace(".jpg", ".txt")
                    # st.write(txtpath)
                    st.header('下为检测后的标签')
                    line_list = []
                    line_list2 = []
                    line_list3 = []
                    with open(txtpath, "r") as f:  # 打开文件
                        for line in f.readlines():
                            # st.write('line', line)
                            line = line.strip('\n')  #去掉列表中每一个元素的换行符
                            if line[:2].strip() == "10":
                                line_list.append("counter")
                            else:
                                line_list.append(line[:2].strip())
                            l2 = line[2:].strip().split()
                            # st.write('l2', l2)
                            line_list2.append(l2[:4])
                            line_list3.append(l2[4])
                            print(line)
                    # st.write('line_list', line_list)
                    # st.write('line_list2', line_list2)
                    # st.write('line_list3', line_list3)
                    df = pd.DataFrame(data=np.zeros((len(line_list), 3)),
                      columns=['Labels', 'Position', 'Confidenc'],
                      index=np.linspace(1, len(line_list), len(line_list), dtype=int))
                    i = 0
                    for (l, p, c) in zip(line_list, line_list2):
                        df.iloc[i,0] = l
                        df.iloc[i,1] = p
                        df.iloc[i,2] = c
                        i += 1
                    html = df.to_html(escape=False)
                    html2 = html.replace('<tr>', '<tr align="center">')
                    html3 = html2.replace('<th>', '<th align="center">')
                    st.write(html3, unsafe_allow_html=True)
                    st.balloons()
            else:
                st.header('下为检测后的视频')
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()
