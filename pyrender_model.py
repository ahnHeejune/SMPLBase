"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

Utility script for rendering the SMPL model using OpenDR.
"""

import numpy as np

from opendr.camera import ProjectPoints
'''
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
'''
import cv2

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


class DummyRenderer(object):

    def __init__(self):
        self.cam = None
        self.frustum = None
        
        


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k
    '''
    rn = ColoredRenderer()
    '''
    rn = DummyRenderer()
    rn.cam = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn
   

def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], 
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def simple_renderer(rn, verts, faces, yrot=np.radians(120)):

    ''' 
      OpenGL 렌더러를 사용하는 대신 OpenCV의 프로젝션을 그냥 사용
    '''
    cam = rn.cam
    cam.v = verts
    
    w = rn.frustum['width']
    h = rn.frustum['height']
    
    if rn.background_image is not None:
        img = rn.background_image.copy()  # use 
    else:
        img = np.zeros((h,w,3))  # float?
    
    # mapping points into the 
    for i in range(cam.r.shape[0]):
        x,y = int(cam.r[i, 0]), int(cam.r[i,1])
        cv2.circle(img, center = (x,y), radius = 2, color = (1.0,0.0,0.0), thickness = -1)
        '''
        if x >= 0 and x < w and y >= 0 and y < h:
            img[y,x, :] = 1.0
        '''
    return img
    
    '''
    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r
    '''


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def render_model(verts, faces, w, h, cam, near=0.5, far=25, img=None):

    """ SMPL 모델을 렌더링 함 """
    
    # 카메라 세팅 
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)
    
    # BG 이미지가 있으면 사용하고, 아니면 white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img  # 픽셀값 범위를 OpenGL 스타일의 (0,1)로 변환 

    imtmp = simple_renderer(rn, verts, faces)

    # If white bg, make transparent.
    if img is None:
        imtmp = get_alpha(imtmp)

    return imtmp
   