from skimage import measure


import SimpleITK as sitk
import numpy as np
import pydicom as dicom
import glob
import os
import pydicom as dicom
from scipy.ndimage import zoom
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cupy as cp
from cupy import sin,cos
from sklearn.metrics import mutual_info_score as mfs
from sklearn.metrics import normalized_mutual_info_score as nmfs

import cv2




class img_reg_2d:
    def __init__(self,fx,mx,fy,my,fz,mz):

        self.fx=fx
        self.mx=mx
        self.fy=fy
        self.my=my
        self.fz=fz
        self.mz=mz

        #根据设置的间距生成坐标索引
        self.downsampel(1,1,1)

        #求解三个重心
        self.ans_x=self.cal_center(self.fx,self.mx)
        self.ans_y=self.cal_center(self.fy,self.my)
        self.ans_z=self.cal_center(self.fz,self.mz)

        self.REX=self.remat(self.ans_x)
        self.REY=self.remat(self.ans_y)
        self.REZ=self.remat(self.ans_z)

        self.globx=self.ans_y[1]
        self.globy=self.ans_x[1]
        self.globz=self.ans_x[2]
        self.glob_ans=cp.asarray([self.globx.copy(),self.globy.copy(),self.globz.copy()])

        self.downsampel(1,1,1)



    def downsampel(self,spacing1,spacing2,spacing3):
        grid_loc_x = []
        for i in range(2):
            grid_loc_x.append(cp.arange(0, int(cp.floor(self.fx.shape[i] / spacing1))*spacing1, spacing1))
        meshgrid = cp.meshgrid(grid_loc_x[0], grid_loc_x[1])
        self.xx, self.xy=meshgrid[0].transpose([1, 0]).reshape(-1, 1),meshgrid[1].transpose([1, 0]).reshape(-1, 1)

        grid_loc_y = []
        for i in range(2):
            grid_loc_y.append(cp.arange(0, int(cp.floor(self.fy.shape[i] / spacing2))*spacing2, spacing2))
        meshgrid = cp.meshgrid(grid_loc_y[0], grid_loc_y[1])
        self.yx, self.yy=meshgrid[0].transpose([1, 0]).reshape(-1, 1),meshgrid[1].transpose([1, 0]).reshape(-1, 1)

        grid_loc_z = []
        for i in range(2):
            grid_loc_z.append(cp.arange(0, int(cp.floor(self.fz.shape[i] / spacing3))*spacing3, spacing3))
        meshgrid = cp.meshgrid(grid_loc_z[0], grid_loc_z[1])
        self.zx, self.zy=meshgrid[0].transpose([1, 0]).reshape(-1, 1),meshgrid[1].transpose([1, 0]).reshape(-1, 1)

        self.down_fx=self.fx[self.xx,self.xy]
        self.down_fy=self.fy[self.yx,self.yy]
        self.down_fz=self.fz[self.zx,self.zy]

    def remat(self,mat):
        theta,x,y=mat.copy()
        theta=theta*cp.pi/180.0
        RE=cp.zeros((3,3))
        RE[0,0]=cp.cos(theta)
        RE[0,1]=-cp.sin(theta)
        RE[0,2]=x
        RE[1,0]=cp.sin(theta)
        RE[1,1]=cp.cos(theta)
        RE[1,2]=y
        RE[2,0]=0
        RE[2,1]=0
        RE[2,2]=1
        return RE

    def cal_center(self,fixed,moving):

       center=cp.zeros(3)

       fixed_center = cp.asarray([0.0, 0.0])
       fixedspacingseries = []
       for i in range(2):
           fixedspacingseries.append(cp.arange(0, fixed.shape[i], 1))
       meshgrid = cp.meshgrid(fixedspacingseries[0], fixedspacingseries[1])
       fixed_center[0] = (meshgrid[0].transpose([1, 0]) * fixed).sum() / fixed.sum()
       fixed_center[1] = (meshgrid[1].transpose([1, 0]) * fixed).sum() / fixed.sum()



       moving_center = cp.asarray([0.0, 0.0])
       movingspacingseries = []
       for i in range(2):
           movingspacingseries.append(cp.arange(0, moving.shape[i], 1))
       meshgrid = cp.meshgrid(movingspacingseries[0], movingspacingseries[1])
       moving_center[0] = (meshgrid[0].transpose([1, 0]) *moving).sum() / moving.sum()
       moving_center[1] = (meshgrid[1].transpose([1, 0]) * moving).sum() / moving.sum()


       center[1:3]=moving_center-fixed_center


       return  center

    #图像变换后进行双线性插值
    def rot_tran_bil_interpolate(self,img,x_ord,y_ord,mat):
        # 旋转 变换是沿着z,x,y方向进行的，先旋转再平移
        rows, cols= img.shape
        img[:,0:2]=0
        img[0:2,:]=0
        img[-2:-1,:]=0
        img[:,-2:-1]=0

        # 最近邻插值
        fnx = mat[0, 0] * x_ord + mat[0, 1] * y_ord + mat[0, 2]
        fny = mat[1, 0] * x_ord + mat[1, 1] * y_ord + mat[1, 2]
        dx = fnx - cp.floor(fnx)
        dy = fny - cp.floor(fny)
        fx1 = cp.clip((cp.floor(fnx)).astype(cp.int32), 0, rows - 2)
        fy1 = cp.clip((cp.floor(fny)).astype(cp.int32), 0, cols - 2)
        fx2 = fx1 + 1
        fy2 = fy1 + 1

        dstimg= (img[fx1, fy1] * (1 - dx) + img[fx2, fy1] * dx) * (1 - dy) \
                             + (img[fx1, fy2] * (1 - dx) + img[fx2, fy2] * dx) * dy

        return dstimg


    def cal_normalize_mutual(self,x, y):
        # https://blog.csdn.net/sihaiyinan/article/details/112196356#:~:text=%E4%BD%BF%E7%94%A8python%E4%B8%AD%E7%9A%84numpy%E5%8C%85%E6%88%96%E8%80%85sklearn%E5%8F%AF%E4%BB%A5%E5%BE%88%E6%96%B9%E4%BE%BF%E7%9A%84%E8%AE%A1%E7%AE%97%E4%BA%92%E4%BF%A1%E6%81%AF%EF%BC%8C%E8%AE%A1%E7%AE%97%E4%BB%A3%E7%A0%81%E5%A6%82%E4%B8%8B%EF%BC%9A%20import%20cv%202,import%20numpy%20as%20np

        size = len(x.reshape(-1))
        hist_x = cp.histogram(x, 256, (0, 255))[0] / size
        hist_y = cp.histogram(y, 256, (0, 255))[0] / size
        entry_x = -cp.sum(hist_x * cp.log(hist_x + 1e-8))
        entry_y = -cp.sum(hist_y * cp.log(hist_y + 1e-8))

        hist_xy = cp.histogram2d(x.reshape(-1), y.reshape(-1), 256, ((0, 255), (0, 255)))[0] / (1.0 * size)
        entry_xy = -cp.sum(hist_xy * cp.log(hist_xy + 1e-8))

        #mutual = 2 * (1 - entry_xy / ((entry_x + entry_y)))
        mutual=(entry_x+entry_y-entry_xy)/min(entry_x,entry_y)

        return mutual

    def optimize(self,iteration,step_iteration,converange=0.01):
        x_iteration=[]
        y_iteration=[]
        z_iteration=[]
        loss_iteration=[]
        self.best_ans=self.glob_ans
        self.flag=0
        converg=1e6

        iterate=iteration
        delt = 0.01
        cong_fore=0.0
        cong_now=0.0
        converg=1
        while (iterate > 0 and converg>converange):
            iterate -= 1

            #计算当前情况下的三对图像互信息
            RE_x = self.remat(self.ans_x)
            tempx= self.rot_tran_bil_interpolate(self.mx, self.xx, self.xy, RE_x)
            mutu_x = self.cal_normalize_mutual(self.down_fx, tempx)
            RE_y = self.remat(self.ans_y)
            tempy= self.rot_tran_bil_interpolate(self.my, self.yx, self.yy, RE_y)
            mutu_y = self.cal_normalize_mutual(self.down_fy, tempy)
            RE_z = self.remat(self.ans_z)
            tempz= self.rot_tran_bil_interpolate(self.mz, self.zx, self.zy, RE_z)
            mutu_z = self.cal_normalize_mutual(self.down_fz, tempz)
            mutu_flag=mutu_x+mutu_y+mutu_z
            #存储最优解


            #计算yz平面的梯度
            ans_yz=self.ans_x.copy()
            ans_yz[1]+=delt
            RE_yz = self.remat(ans_yz)
            tempyz= self.rot_tran_bil_interpolate(self.mx, self.xx, self.xy, RE_yz)
            mutu_yz = self.cal_normalize_mutual(self.down_fx, tempyz)
            grd_yz_y=(mutu_yz-mutu_x)/delt

            ans_yz=self.ans_x.copy()
            ans_yz[2]+=delt
            RE_yz = self.remat(ans_yz)
            tempyz= self.rot_tran_bil_interpolate(self.mx, self.xx, self.xy ,RE_yz)
            mutu_yz = self.cal_normalize_mutual(self.down_fx, tempyz)
            grd_yz_z=(mutu_yz-mutu_x)/delt


            #计算xz平面的梯度
            ans_xz=self.ans_y.copy()
            ans_xz[1]+=delt
            RE_xz = self.remat(ans_xz)
            tempxz= self.rot_tran_bil_interpolate(self.my, self.yx, self.yy, RE_xz)
            mutu_xz = self.cal_normalize_mutual(self.down_fy, tempxz)
            grd_xz_x=(mutu_xz-mutu_y)/delt

            ans_xz=self.ans_y.copy()
            ans_xz[2]+=delt
            RE_xz = self.remat(ans_xz)
            tempxz= self.rot_tran_bil_interpolate(self.my, self.yx, self.yy, RE_xz)
            mutu_xz = self.cal_normalize_mutual(self.down_fy, tempxz)
            grd_xz_z=(mutu_xz-mutu_y)/delt


            #计算xy平面的梯度
            ans_xy=self.ans_z.copy()
            ans_xy[1]+=delt
            RE_xy = self.remat(ans_xy)
            tempxy= self.rot_tran_bil_interpolate(self.mz, self.zx, self.zy, RE_xy)
            mutu_xy = self.cal_normalize_mutual(self.down_fz, tempxy)
            grd_xy_x=(mutu_xy-mutu_z)/delt
            """修改处"""

            ans_xy=self.ans_z.copy()
            ans_xy[2]+=delt
            RE_xy = self.remat(ans_xy)
            tempxy= self.rot_tran_bil_interpolate(self.mz, self.zx, self.zy, RE_xy)
            mutu_xy = self.cal_normalize_mutual(self.down_fz, tempxy)
            grd_xy_y=(mutu_xy-mutu_z)/delt
            """修改处"""
            grdx=grd_xy_x+grd_xz_x
            grdy=grd_xy_y+grd_yz_y
            grdz=grd_xz_z+grd_yz_z
            grd=cp.asarray([grdx,grdy,grdz])

            step = 10 / cp.max(cp.abs(grd))
            flag = []
            step_arr = step * cp.logspace(0, step_iteration-1, step_iteration, base=0.5)
            for i in step_arr:

                glob_copy=self.glob_ans.copy()
                glob_copy+=grd*i
                #更新配准矩阵
                ans_x_copy=self.ans_x.copy()
                ans_y_copy=self.ans_y.copy()
                ans_z_copy=self.ans_z.copy()
                ans_x_copy[1:3] = cp.asarray([glob_copy[1], glob_copy[2]])
                ans_y_copy[1:3] = cp.asarray([glob_copy[0], glob_copy[2]])
                ans_z_copy[1:3] = cp.asarray([glob_copy[0], glob_copy[1]])

                #计算图像互信息
                RE_x = self.remat(ans_x_copy)
                tempx = self.rot_tran_bil_interpolate(self.mx, self.xx, self.xy, RE_x)
                mutu_x = self.cal_normalize_mutual(self.down_fx, tempx)

                RE_y =self.remat(ans_y_copy)
                tempy = self.rot_tran_bil_interpolate(self.my, self.yx, self.yy, RE_y)
                mutu_y = self.cal_normalize_mutual(self.down_fy, tempy)

                RE_z =self.remat(ans_z_copy)
                tempz = self.rot_tran_bil_interpolate(self.mz, self.zx, self.zy, RE_z)
                mutu_z = self.cal_normalize_mutual(self.down_fz, tempz)

                #互信息相加
                mutu_step = mutu_x + mutu_y + mutu_z
                flag.append(mutu_step.copy())
            flag = cp.asarray(flag)
            best_loc = cp.where(flag == flag.max())
            step_forward=step_arr[best_loc]*grd
            self.glob_ans+=step_forward

            #存储迭代后的最优解
            if flag[best_loc]>self.flag:
                self.flag=flag[best_loc].copy()
                self.best_ans=self.glob_ans.copy()

            #收敛条件，改变量过小，停止计算
            cong_fore=cong_now
            cong_now=cp.sqrt(cp.dot(step_forward,step_forward.T))
            converg=cp.abs(cong_now-cong_fore)
            print("第",iteration-iterate,"次迭代结果为：",self.glob_ans)
            self.ans_x[1:3]=cp.asarray([self.glob_ans[1],self.glob_ans[2]])
            self.ans_y[1:3]=cp.asarray([self.glob_ans[0],self.glob_ans[2]])
            self.ans_z[1:3]=cp.asarray([self.glob_ans[0],self.glob_ans[1]])

class multmode_regist:
    def __init__(self,path):

        #输入路径
        self.path=path
        #读取当前图像
        self.fixed,self.moving=self.read_image()


        #初始化配准参数和配准矩阵,默认为根据两幅图像的重心进行初始化
        self.ans=cp.zeros(6)
        self.ans[3:6]=self.cal_center(self.moving)-self.cal_center(self.fixed)
        self.RE=self.REmat(self.ans)
        #确立旋转中心，默认为原点，后期可自行更改
        self.rota_cent=cp.asarray([0.0,0.0,0.0])
        #默认下采样系数为1，即不向下均匀采样，同时根据采样情况生成坐标索引
        self.down_sample(1)


    #读取图像数据以及重心
    def read_image(self):
        # Read fixed Images 的存储路径
        dicom_series_fixed = glob.glob(os.path.join(self.path,'CT', '*.dcm'))
        clices = []
        for s in dicom_series_fixed:
            clices.append(dicom.read_file(s, force=True))
        clices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        image_fixed = np.stack([s.pixel_array for s in clices], axis=-1)

        clice = dicom.read_file(dicom_series_fixed[0])
        sp_fixed = clice.PixelSpacing  # voxel spacing
        fixed_thickness = np.asarray(clice.SliceThickness)
        # print(spct)

        # Read fixed Images 的存储路径
        dicom_series_moving = glob.glob(os.path.join(self.path,'MR', '*.dcm'))
        clices = []
        for s in dicom_series_moving:
            clices.append(dicom.read_file(s, force=True))
        clices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        image_mov = np.stack([s.pixel_array for s in clices], axis=-1)

        clice = dicom.read_file(dicom_series_moving[0])
        sp_moving = clice.PixelSpacing  # voxel spacing
        moving_thickness=np.asarray(clice.SliceThickness)
        image_moving = zoom(image_mov, (sp_moving[0] / sp_fixed[0], sp_moving[1] / sp_fixed[1], moving_thickness/fixed_thickness))

        rows_max,cols_max,slice_max=max(image_fixed.shape[0],image_moving.shape[0]),max(image_fixed.shape[1],image_moving.shape[1]),max(image_fixed.shape[2],image_moving.shape[2])
        fixed=np.zeros((rows_max,cols_max,slice_max))
        moving=np.zeros((rows_max,cols_max,slice_max))
        fixed[:image_fixed.shape[0],:image_fixed.shape[1],:image_fixed.shape[2]]=image_fixed
        moving[:image_moving.shape[0],:image_moving.shape[1],:image_moving.shape[2]]=image_moving
        fixed = 255 * (fixed - fixed.min()) / (fixed.max() - fixed.min())
        moving = 255 * (moving - moving.min()) / (moving.max() - moving.min())
        return cp.asarray(fixed),cp.asarray(moving)
    #计算图像重心
    def cal_center(self,img):


       cente = cp.asarray([0.0, 0.0, 0.0])
       spacing_series = []
       for i in range(3):
           spacing_series.append(cp.arange(0, img.shape[i], 1))
       meshgrid = cp.meshgrid(spacing_series[0], spacing_series[1], spacing_series[2])
       cente[0] = (meshgrid[0].transpose([1, 0, 2]) * img).sum() / img.sum()
       cente[1] = (meshgrid[1].transpose([1, 0, 2]) * img).sum() / img.sum()
       cente[2] = (meshgrid[2].transpose([1, 0, 2]) * img).sum() / img.sum()

       return  cente

    def down_sample(self, spacing):
        grid_loc = []
        for i in range(3):
            grid_loc.append(cp.arange(0, int(cp.floor(self.fixed.shape[i] / spacing))*spacing, spacing))
        meshgrid = cp.meshgrid(grid_loc[0], grid_loc[1],
                               grid_loc[2])  # 对于Meshgrid的使用还是不太清楚,具体看https://zhuanlan.zhihu.com/p/105148901
        self.x_ord, self.y_ord, self.z_ord = meshgrid[0].transpose([1, 0, 2]).reshape(-1, 1), \
                                          meshgrid[1].transpose([1, 0, 2]).reshape(-1, 1), \
                                          meshgrid[2].transpose([1, 0, 2]).reshape(-1, 1)


        #存储抽样后的图像，为梯度计算做准备
        self.down_fixed=self.fixed[self.x_ord,self.y_ord,self.z_ord]
    #根据6自由度初始化配准矩阵
    def REmat(self,arr):
        alp, bet, gam, x, y, z = arr
        RE = cp.zeros([4, 4])
        alp = alp * cp.pi / 180.0
        bet = bet * cp.pi / 180.0
        gam = gam * cp.pi / 180.0
        RE[0][0] = cp.cos(gam) * cp.cos(bet) - cp.sin(alp) * cp.sin(bet) * cp.sin(gam)
        RE[0][1] = -cp.sin(gam) * cp.cos(alp)
        RE[0][2] = cp.sin(gam) * cp.sin(alp) * cp.cos(bet) + cp.sin(bet) * cp.cos(gam)
        RE[0][3] = x
        RE[1][0] = cp.sin(gam) * cp.cos(bet) + cp.sin(alp) * cp.sin(bet) * cp.cos(gam)
        RE[1][1] = cp.cos(gam) * cp.cos(alp)
        RE[1][2] = -cp.cos(gam) * cp.sin(alp) * cp.cos(bet) + cp.sin(gam) * cp.sin(bet)
        RE[1][3] = y
        RE[2][0] = -cp.cos(alp) * cp.sin(bet)
        RE[2][1] = cp.sin(alp)
        RE[2][2] = cp.cos(bet) * cp.cos(alp)
        RE[2][3] = z
        RE[3][0] = 0
        RE[3][1] = 0
        RE[3][2] = 0
        RE[3][3] = 1
        return RE

    #图像变换后进行双线性插值
    def rot_tran_bil_interpolate(self,img,x_ord,y_ord,z_ord,mat):
        # 旋转 变换是沿着z,x,y方向进行的，先旋转再平移
        rows, cols, slic = img.shape
        img[:,:,0]=0
        img[:,0,:]=0
        img[0,:,:]=0


        # 最近邻插值
        fnx = mat[0, 0] * (x_ord-self.rota_cent[0]) + mat[0, 1] * (y_ord-self.rota_cent[1]) + mat[0, 2] * (z_ord-self.rota_cent[2]) + mat[0, 3]+self.rota_cent[0]
        fny = mat[1, 0] * (x_ord-self.rota_cent[0]) + mat[1, 1] * (y_ord-self.rota_cent[1]) + mat[1, 2] * (z_ord-self.rota_cent[2]) + mat[1, 3]+self.rota_cent[1]
        fnz = mat[2, 0] * (x_ord-self.rota_cent[0]) + mat[2, 1] * (y_ord-self.rota_cent[1]) + mat[2, 2] * (z_ord-self.rota_cent[2]) + mat[2, 3]+self.rota_cent[2]


        dx = fnx - cp.floor(fnx)
        dy = fny - cp.floor(fny)
        dz = fnz - cp.floor(fnz)
        fx1 = cp.clip((cp.floor(fnx)).astype(cp.int32), 0, rows - 2)
        fy1 = cp.clip((cp.floor(fny)).astype(cp.int32), 0, cols - 2)
        fz1 = cp.clip((cp.floor(fnz)).astype(cp.int32), 0, slic - 2)
        fx2 = fx1 + 1
        fy2 = fy1 + 1
        fz2 = fz1 + 1

        # 先在z1层进行二维双线性插值
        z1 = (img[fx1, fy1, fz1] * (1 - dx) + img[fx2, fy1, fz1] * dx) * (1 - dy) \
             + (img[fx1, fy2, fz1] * (1 - dx) + img[fx2, fy2, fz1] * dx) * dy
        # z2层次进行二维双线性插值
        z2 = (img[fx1, fy1, fz2] * (1 - dx) + img[fx2, fy1, fz2] * dx) * (1 - dy) \
             + (img[fx1, fy2, fz2] * (1 - dx) + img[fx2, fy2, fz2] * dx) * dy

        dstimg = z1 * (1 - dz) + z2 * dz
        return dstimg


    def cal_normalize_mutual(self,x, y):
        # https://blog.csdn.net/sihaiyinan/article/details/112196356#:~:text=%E4%BD%BF%E7%94%A8python%E4%B8%AD%E7%9A%84numpy%E5%8C%85%E6%88%96%E8%80%85sklearn%E5%8F%AF%E4%BB%A5%E5%BE%88%E6%96%B9%E4%BE%BF%E7%9A%84%E8%AE%A1%E7%AE%97%E4%BA%92%E4%BF%A1%E6%81%AF%EF%BC%8C%E8%AE%A1%E7%AE%97%E4%BB%A3%E7%A0%81%E5%A6%82%E4%B8%8B%EF%BC%9A%20import%20cv%202,import%20numpy%20as%20np

        size = len(x.reshape(-1))
        hist_x = cp.histogram(x, 256, (0, 255))[0] / size
        hist_y = cp.histogram(y, 256, (0, 255))[0] / size
        entry_x = -cp.sum(hist_x * cp.log(hist_x + 1e-8))
        entry_y = -cp.sum(hist_y * cp.log(hist_y + 1e-8))

        hist_xy = cp.histogram2d(x.reshape(-1), y.reshape(-1), 256, ((0, 255), (0, 255)))[0] / (1.0 * size)
        entry_xy = -cp.sum(hist_xy * cp.log(hist_xy + 1e-8))

        mutual = 2 * (1 - entry_xy / ((entry_x + entry_y)))

        return mutual

    def cal_grad(self, count, delt, mutulfs,ans,fixed,moving,x_ord,y_ord,z_ord):
        """

        Parameters
        ----------
        vetor:初始化6自由度
        count：自由度序号
        delt：数值解计算的偏移量
        fixed：fixed图像
        moving：moving图像
        indx：x方向坐标索引
        indy：y方向坐标索引
        indz：z方向坐标索引
        mutulfs：初始时刻互信息，用于计算梯度

        Returns
        -------
        grd:计算得到的梯度

        """

        t_ans = ans.copy()
        t_ans[count] += delt
        # 计算RE矩阵
        alp, bet, gam, x, y, z = t_ans

        mat = cp.zeros([4, 4])
        alp = alp * cp.pi / 180.0
        bet = bet * cp.pi / 180.0
        gam = gam * cp.pi / 180.0
        mat[0][0] = cp.cos(gam) * cp.cos(bet) - cp.sin(alp) * cp.sin(bet) * cp.sin(gam)
        mat[0][1] = -cp.sin(gam) * cp.cos(alp)
        mat[0][2] = cp.sin(gam) * cp.sin(alp) * cp.cos(bet) + cp.sin(bet) * cp.cos(gam)
        mat[0][3] = x
        mat[1][0] = cp.sin(gam) * cp.cos(bet) + cp.sin(alp) * cp.sin(bet) * cp.cos(gam)
        mat[1][1] = cp.cos(gam) * cp.cos(alp)
        mat[1][2] = -cp.cos(gam) * cp.sin(alp) * cp.cos(bet) + cp.sin(gam) * cp.sin(bet)
        mat[1][3] = y
        mat[2][0] = -cp.cos(alp) * cp.sin(bet)
        mat[2][1] = cp.sin(alp)
        mat[2][2] = cp.cos(bet) * cp.cos(alp)
        mat[2][3] = z
        mat[3][0] = 0
        mat[3][1] = 0
        mat[3][2] = 0
        mat[3][3] = 1

        # 计算根据旋转矩阵的插值结果
        rows, cols, slic = moving.shape
        moving[:,:,0]=0
        moving[:,0,:]=0
        moving[0,:,:]=0

        # 最近邻插值
        fnx = mat[0, 0] * (x_ord-self.rota_cent[0]) + mat[0, 1] * (y_ord-self.rota_cent[1]) + mat[0, 2] * (z_ord-self.rota_cent[2]) + mat[0, 3]+self.rota_cent[0]
        fny = mat[1, 0] * (x_ord-self.rota_cent[0]) + mat[1, 1] * (y_ord-self.rota_cent[1]) + mat[1, 2] * (z_ord-self.rota_cent[2]) + mat[1, 3]+self.rota_cent[1]
        fnz = mat[2, 0] * (x_ord-self.rota_cent[0]) + mat[2, 1] * (y_ord-self.rota_cent[1]) + mat[2, 2] * (z_ord-self.rota_cent[2]) + mat[2, 3]+self.rota_cent[2]

        dx = fnx - cp.floor(fnx)
        dy = fny - cp.floor(fny)
        dz = fnz - cp.floor(fnz)
        fx1 = cp.clip((cp.floor(fnx)).astype(cp.int32), 0, rows - 2)
        fy1 = cp.clip((cp.floor(fny)).astype(cp.int32), 0, cols - 2)
        fz1 = cp.clip((cp.floor(fnz)).astype(cp.int32), 0, slic - 2)
        fx2 = fx1 + 1
        fy2 = fy1 + 1
        fz2 = fz1 + 1

        # 先在z1层进行二维双线性插值
        z1 = (moving[fx1, fy1, fz1] * (1 - dx) + moving[fx2, fy1, fz1] * dx) * (1 - dy) \
             + (moving[fx1, fy2, fz1] * (1 - dx) + moving[fx2, fy2, fz1] * dx) * dy
        # z2层次进行二维双线性插值
        z2 = (moving[fx1, fy1, fz2] * (1 - dx) + moving[fx2, fy1, fz2] * dx) * (1 - dy) \
             + (moving[fx1, fy2, fz2] * (1 - dx) + moving[fx2, fy2, fz2] * dx) * dy

        moving_iter= z1 * (1 - dz) + z2 * dz

        size = len(fixed.reshape(-1))
        hist_x = cp.histogram(fixed, 256, (0, 255))[0] / size
        hist_y = cp.histogram(moving_iter, 256, (0, 255))[0] / size
        entry_x = -cp.sum(hist_x * cp.log(hist_x + 1e-8))
        entry_y = -cp.sum(hist_y * cp.log(hist_y + 1e-8))

        hist_xy = cp.histogram2d(fixed.reshape(-1), moving_iter.reshape(-1), 256, ((0, 255), (0, 255)))[0] / (
                    1.0 * size)
        entry_xy = -cp.sum(hist_xy * cp.log(hist_xy + 1e-8))

        mutual = 2 * (1 - entry_xy / ((entry_x + entry_y)))
        grd = (mutual - mutulfs) / delt
        return grd

    #一阶优化
    def optimize(self,iteration, step_cal_iter,convergance,rota_center="ORIGIN"):
        """

        Parameters
        ----------
        iteration
        step_cal_iter
        convergance
        rota_center=ORIGIN,GEOMETRY,MOMENTS

        Returns
        -------

        """
        if rota_center=="ORIGIN":
            self.rota_cent=cp.asarray([0.0,0.0,0.0])
        if rota_center=="GEOMETRY":
            self.rota_cent=cp.asarray([(self.moving.shape[0]-1) / 2.0, (self.moving.shape[1]-1) / 2.0, (self.moving.shape[2]-1) / 2.0])
        if rota_center=="MOMENTS":
            self.rota_cent=self.cal_center(self.moving)
        self.flag = 0
        iterat=iteration

        converge=1e6
        cong_fore=0.0
        cong_now=0.0
        while (iterat > 0 and converge>convergance):
            iterat -= 1

            del_theta = 1e-3
            del_mov =1e-4

            temp_comp = self.rot_tran_bil_interpolate(self.moving, self.x_ord, self.y_ord, self.z_ord, self.RE)
            mutulfs = self.cal_normalize_mutual(self.down_fixed, temp_comp)



            grda = self.cal_grad( 0, del_theta, mutulfs,self.ans,self.down_fixed,self.moving,self.x_ord,self.y_ord,self.z_ord)
            grdb = self.cal_grad( 1, del_theta, mutulfs,self.ans,self.down_fixed,self.moving,self.x_ord,self.y_ord,self.z_ord)
            grdg = self.cal_grad( 2, del_theta, mutulfs,self.ans,self.down_fixed,self.moving,self.x_ord,self.y_ord,self.z_ord)
            grdx = self.cal_grad( 3, del_mov, mutulfs,self.ans,self.down_fixed,self.moving,self.x_ord,self.y_ord,self.z_ord)
            grdy = self.cal_grad( 4, del_mov, mutulfs,self.ans,self.down_fixed,self.moving,self.x_ord,self.y_ord,self.z_ord)
            grdz = self.cal_grad( 5, del_mov, mutulfs,self.ans,self.down_fixed,self.moving,self.x_ord,self.y_ord,self.z_ord)
            #grda,grdb,grdg=cp.zeros(3)
            grd = cp.asarray([grda, grdb, grdg, grdx, grdy, grdz])

            gradmax = cp.max(cp.abs(grd))
            step = 10 / gradmax
            # 生成步长矩阵

            flag = []
            step_arr = step * cp.logspace(0, step_cal_iter - 1, step_cal_iter, base=0.5)
            for i in step_arr:
                ans_copy = self.ans.copy()
                ans_copy += i * grd
                Re = self.REmat(ans_copy)
                tempp = self.rot_tran_bil_interpolate(self.moving,self.x_ord,self.y_ord,self.z_ord,Re)
                flag.append(self.cal_normalize_mutual(self.down_fixed, tempp))
            flag = cp.asarray(flag)
            best_loc = cp.where(flag == flag.max())

            #确定前进距离
            forward_choose=step_arr[best_loc] * grd
            self.ans+= forward_choose
            cong_fore=cong_now
            cong_now=cp.sqrt(cp.dot(forward_choose,forward_choose.T))
            converge=cp.abs(cong_now-cong_fore)
            # 存储最优解
            if flag[best_loc] > self.flag:
                self.best_ans = self.ans.copy()
                self.save_count = iteration - iterat
                self.flag= flag[best_loc].copy()

            print("第",iteration-iterat,"次迭代结果为：",self.ans)
            self.RE = self.REmat(self.ans)




def regis_based_sitk(CT,MR,iteration,learning_rate):
       fixed=np.transpose(CT,[2,0,1])
       moving=np.transpose(MR,[2,0,1])

       # 直接从numpy矩阵读取为dcm格式文件
       fixed_dcm = sitk.GetImageFromArray(fixed, isVector=None)
       moving_dcm = sitk.GetImageFromArray(moving, isVector=None)

       #将dcm文件转化为从float32型，simpleitk只能处理float32和float64位的数据
       fixed_dcm = sitk.Cast(fixed_dcm, sitk.sitkFloat32)  # 常见的软件要求pixel value为float型,常见的医学图像处理软件只能处理float型的图像
       moving_dcm = sitk.Cast(moving_dcm, sitk.sitkFloat32)


       """http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/62_Registration_Tuning.html"""
       """此行用于初始化transform的旋转中心和平移量，主要是根据moving图像和fixed图像二者之间的中心差值作为旋转中心以及对应的平移量"""
       """这个transform是用来将fixed,moving移动到理想位置，
       https://stackoverflow.com/questions/38701242/itk-difference-betweeen-setinitialtransform-and-setmovinginitialtransform"""
       initial_transform = sitk.CenteredTransformInitializer(fixed_dcm,
                                                             moving_dcm,
                                                             sitk.Euler3DTransform(),
                                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)#sitk.CenteredTransformInitializerFilter.MOMENT



       """transform初始化"""
       registration_method = sitk.ImageRegistrationMethod()

       # Similarity metric settings.
       """设置相似性度量准则"""
       registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
       """设置相似性度量准则的采样策略NONE,REGULAR,RANDOM，这里是不随机采样"""
       registration_method.SetMetricSamplingStrategy(registration_method.NONE)
       """设置采样率"""
       #registration_method.SetMetricSamplingPercentage(0.1)
       """设置图像变换的插值方式"""
       registration_method.SetInterpolator(sitk.sitkLinear)

       """# Optimizer settings.,设置优化器梯度下降参数，包括学习率，最大迭代次数，收敛"""
       registration_method.SetOptimizerAsGradientDescent(learningRate=learning_rate, numberOfIterations=iteration)#,
                                                         #convergenceMinimumValue=1e-16, convergenceWindowSize=512)

       """根据参数变化在物理空间引起的最大体素偏移，估计transform的前进尺度"""
       registration_method.SetOptimizerScalesFromPhysicalShift()

       """"# Setup for the multi-resolution framework.,设置多尺度配准的缩放尺度和平滑因子"""
       #registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
       #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
       """保障每层每个体素都能被平滑因子作用到"""
       #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

       # Set the initial moving and optimized transforms.
       """迭代过程中使用到的优化变换"""
       optimized_transform = sitk.Euler3DTransform()
       """对moving_initialtransform做初始化，移动到理想的位置开始进行配准"""
       registration_method.SetMovingInitialTransform(initial_transform)#这两行看起来像是更改旋转中心
       registration_method.SetFixedInitialTransform(initial_transform)
       registration_method.SetInitialTransform(optimized_transform, inPlace=False)

       # Need to compose the transformations after registration.
       outTx = registration_method.Execute(fixed_dcm, moving_dcm)
       # final_transform_v4.AddTransform(initial_transform)
       print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
       print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
       print(outTx)
       if ("SITK_NOSHOW" not in os.environ):
           resampler = sitk.ResampleImageFilter()
           resampler.SetReferenceImage(fixed_dcm)
           resampler.SetInterpolator(sitk.sitkLinear)
           resampler.SetDefaultPixelValue(1)
           resampler.SetTransform(outTx)

           out = resampler.Execute(moving_dcm)

       f=sitk.GetArrayFromImage(fixed_dcm)
       mo=sitk.GetArrayFromImage(out)
       f=f.transpose([1,2,0])
       mo=mo.transpose([1,2,0])
       return [cp.asarray(f),cp.asarray(mo)]

#绘制配准后的透视图
def view_comp(imga, imgb,imgc,imgd, degree,i):
    tempa = np.zeros((imga.shape[0], imga.shape[1], 3))
    tempb = np.zeros((imgb.shape[0], imgb.shape[1], 3))
    tempc = np.zeros((imgc.shape[0], imgc.shape[1], 3))
    tempd = np.zeros((imgd.shape[0], imgd.shape[1], 3))
    tempa[:, :, 0] = imga
    tempb[:, :, 1] = imgb
    tempc[:, :, 0] = imgc
    tempd[:, :, 1] = imgd

    overlapping_1 = cv2.addWeighted(tempa, degree, tempb, degree, 0)
    overlapping_2 = cv2.addWeighted(tempc, degree, tempd, degree, 0)
    plt.subplot(121)
    plt.imshow(overlapping_1)
    plt.title(i)
    plt.subplot(122)
    plt.imshow(overlapping_2)
    plt.title(i)
    plt.show()
#读取图像数据以及重心
def read_image(path):
        # Read fixed Images 的存储路径
        dicom_series_fixed = glob.glob(os.path.join(path,'CT', '*.dcm'))
        clices = []
        for s in dicom_series_fixed:
            clices.append(dicom.read_file(s, force=True))
        clices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        image_fixed = np.stack([s.pixel_array for s in clices], axis=-1)

        clice = dicom.read_file(dicom_series_fixed[0])
        sp_fixed = clice.PixelSpacing  # voxel spacing
        fixed_thickness = np.asarray(clice.SliceThickness)
        # print(spct)

        # Read fixed Images 的存储路径
        dicom_series_moving = glob.glob(os.path.join(path,'MR', '*.dcm'))
        clices = []
        for s in dicom_series_moving:
            clices.append(dicom.read_file(s, force=True))
        clices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        image_mov = np.stack([s.pixel_array for s in clices], axis=-1)

        clice = dicom.read_file(dicom_series_moving[0])
        sp_moving = clice.PixelSpacing  # voxel spacing
        moving_thickness=np.asarray(clice.SliceThickness)
        image_moving = zoom(image_mov, (sp_moving[0] / sp_fixed[0], sp_moving[1] / sp_fixed[1], moving_thickness/fixed_thickness))

        rows_max,cols_max,slice_max=max(image_fixed.shape[0],image_moving.shape[0]),max(image_fixed.shape[1],image_moving.shape[1]),max(image_fixed.shape[2],image_moving.shape[2])
        fixed=np.zeros((rows_max,cols_max,slice_max))
        moving=np.zeros((rows_max,cols_max,slice_max))
        fixed[:image_fixed.shape[0],:image_fixed.shape[1],:image_fixed.shape[2]]=image_fixed
        moving[:image_moving.shape[0],:image_moving.shape[1],:image_moving.shape[2]]=image_moving
        fixed = 255 * (fixed - fixed.min()) / (fixed.max() - fixed.min())
        moving = 255 * (moving - moving.min()) / (moving.max() - moving.min())
        return cp.asarray(fixed),cp.asarray(moving)

#去除CT病床
def remove_tab(img):
        img = cp.asnumpy(img.astype(np.uint8))
        # 存储滤波结果
        canny_edge = []
        # 存储二值化结果
        blur = []
        # 滤波后提取边缘
        for i in range(0, img.shape[2]):
            blur.append(cv2.GaussianBlur(img[:, :, i], (15, 15), 0))
            canny_edge.append(cv2.Canny(blur[i], 5, 5))

        # 存储图像中存在的直线
        line_save = []
        threshold, minlen, maxgap, theta = 50, 50, 5, 2
        for i in range(len(canny_edge)):
            edges = canny_edge[i]
            lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180, threshold,
                                    minLineLength=minlen, maxLineGap=maxgap)
            line_temp = []
            if lines is None:
                line_save.append(line_temp)
                continue

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (abs(y1 - y2)) > np.tan(theta * np.pi / 180.0) * abs(x1 - x2):
                    continue
                line_temp.append([x1, y1, x2, y2])
            if line_temp != None:
                line_temp = sorted(line_temp, key=lambda x: x[1], reverse=False)
            line_save.append(line_temp)
        temp = 0
        for i in range(len(canny_edge)):
            line = line_save[i]
            if line == []:
                continue
            line_low = min(line[0][1], line[0][3])
            if line_low > temp:
                temp = line_low
        # tt=min(line_save[11][0][1], line_save[11][0][3])
        img[370:, :, :] *= 0
        return cp.asarray(img)

#计算互信息
def cal_mutual(x, y):
        # https://blog.csdn.net/sihaiyinan/article/details/112196356#:~:text=%E4%BD%BF%E7%94%A8python%E4%B8%AD%E7%9A%84numpy%E5%8C%85%E6%88%96%E8%80%85sklearn%E5%8F%AF%E4%BB%A5%E5%BE%88%E6%96%B9%E4%BE%BF%E7%9A%84%E8%AE%A1%E7%AE%97%E4%BA%92%E4%BF%A1%E6%81%AF%EF%BC%8C%E8%AE%A1%E7%AE%97%E4%BB%A3%E7%A0%81%E5%A6%82%E4%B8%8B%EF%BC%9A%20import%20cv%202,import%20numpy%20as%20np

        size = len(x.reshape(-1))
        hist_x = cp.histogram(x, 256, (0, 255))[0] / size
        loc_x=cp.where(hist_x==0)
        hist_y = cp.histogram(y, 256, (0, 255))[0] / size
        loc_y=cp.where(hist_y==0)
        log_x = hist_x * cp.log(hist_x)
        log_y = hist_y * cp.log(hist_y)
        log_x[loc_x]=0
        log_y[loc_y]=0
        entry_x=-cp.sum(log_x)
        entry_y=-cp.sum(log_y)

        hist_xy = cp.histogram2d(x.reshape(-1), y.reshape(-1), 256, ((0, 255), (0, 255)))[0] / (1.0 * size)
        loc_xy=cp.where(hist_xy==0)
        log_xy = hist_xy * cp.log(hist_xy)
        log_xy[loc_xy]=0
        entry_xy=-cp.sum(log_xy)
        mutual = 2 * (1 - entry_xy / ((entry_x + entry_y)))

        return mutual

def cal_normalize_mutual(x, y):
        # https://blog.csdn.net/sihaiyinan/article/details/112196356#:~:text=%E4%BD%BF%E7%94%A8python%E4%B8%AD%E7%9A%84numpy%E5%8C%85%E6%88%96%E8%80%85sklearn%E5%8F%AF%E4%BB%A5%E5%BE%88%E6%96%B9%E4%BE%BF%E7%9A%84%E8%AE%A1%E7%AE%97%E4%BA%92%E4%BF%A1%E6%81%AF%EF%BC%8C%E8%AE%A1%E7%AE%97%E4%BB%A3%E7%A0%81%E5%A6%82%E4%B8%8B%EF%BC%9A%20import%20cv%202,import%20numpy%20as%20np

        size = len(x.reshape(-1))
        hist_x = cp.histogram(x, 256, (0, 255))[0] / size
        hist_y = cp.histogram(y, 256, (0, 255))[0] / size
        entry_x = -cp.sum(hist_x * cp.log(hist_x + 1e-8))
        entry_y = -cp.sum(hist_y * cp.log(hist_y + 1e-8))

        hist_xy = cp.histogram2d(x.reshape(-1), y.reshape(-1), 256, ((0, 255), (0, 255)))[0] / (1.0 * size)
        entry_xy = -cp.sum(hist_xy * cp.log(hist_xy + 1e-8))

        mutual = 2 * (1 - entry_xy / ((entry_x + entry_y)))

        return mutual

if __name__=="__main__":
    subject_folder = "/home/xuchacha/GUOKE/DIPY_TEST/2d_dataset/5_"
    #subject_folder="/home/xuchacha/GUOKE/DIPY_TEST/2d_dataset/V2019111507"
    fixed,moving=read_image(subject_folder)
    fixed=remove_tab(fixed)


    #基于 SITK的刚性配准结果以及对应的互信息值
    st=time.time()
    fixed_dcm,moving_dcm=regis_based_sitk(cp.asnumpy(fixed),cp.asnumpy(moving),2,0.1)
    en=time.time()
    cost_1=en-st
    #msitk=nmfs(cp.asnumpy(fixed).reshape(-1),cp.asnumpy(moving_dcm).reshape(-1))
    #mnsitk=cal_normalize_mutual(fixed_dcm,moving_dcm)


    #二维粗配准
    fx=cp.sum(fixed,axis=0)
    mx=cp.sum(moving,axis=0)
    fy=cp.sum(fixed,axis=1)
    my=cp.sum(moving,axis=1)
    fz=cp.sum(fixed,axis=2)
    mz=cp.sum(moving,axis=2)
    fx = 255 * (fx - fx.min()) / (fx.max() - fx.min())
    fy = 255 * (fy - fy.min()) / (fy.max() - fy.min())
    fz = 255 * (fz - fz.min()) / (fz.max() - fz.min())
    mx = 255 * (mx - mx.min()) / (mx.max() - mx.min())
    my = 255 * (my - my.min()) / (my.max() - my.min())
    mz = 255 * (mz - mz.min()) / (mz.max() - mz.min())

    #配准对x方向叠加的结果
    reg_2d=img_reg_2d(fx, mx, fy, my, fz, mz)
    reg_3d=multmode_regist(subject_folder)
    reg_3d.fixed=fixed

    #均匀下采样以及多尺度功能

    #reg_2d.glob_ans=reg_2d.best_ans
    reg_2d.downsampel(1,1,1)
    reg_2d.optimize(iteration=20, step_iteration=10, converange=1e-5)
    reg_3d.ans[3:6]=reg_2d.best_ans
    reg_3d.RE=reg_3d.REmat(reg_3d.ans)
    reg_3d.down_sample(1)
    st=time.time()
    reg_3d.optimize(iteration=10, step_cal_iter=10, convergance=1e-6,rota_center="GEOMETRY")
    en=time.time()
    cost_2=en-st

    reg_3d.RE=reg_3d.REmat(reg_3d.best_ans)
    moving_3d=reg_3d.rot_tran_bil_interpolate(reg_3d.moving, reg_3d.x_ord, reg_3d.y_ord, reg_3d.z_ord, reg_3d.RE)
    mutul_self=cal_mutual(reg_3d.fixed, moving_3d)
    #mself=nmfs(cp.asnumpy(reg_3d.fixed).reshape(-1),cp.asnumpy(moving_3d).reshape(-1))
    a=cp.asnumpy(fixed_dcm)
    b=cp.asnumpy(moving_dcm)
    c=cp.asnumpy(reg_3d.fixed)
    d=cp.asnumpy(moving_3d)
    for i in range(0,fixed_dcm.shape[2],4):
        view_comp(a[:,:,i],b[:,:,i],c[:,:,i],d[:,:,i],0.01,i)
    abroke=1





