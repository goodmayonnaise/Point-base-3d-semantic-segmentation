#!/usr/bin/env python3
import numpy as np
import cv2

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, front=False, crop_size=None):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.front = front
    self.crop_size = crop_size

    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()
  
  def read_calib(self, calib_path):
      """
      :param calib_path: Path to a calibration text file.
      :return: dict with calibration matrices.
      """
      calib_all = {}
      with open(calib_path, 'r') as f:
          for line in f.readlines():
              if line == '\n':
                  break
              key, value = line.split(':', 1)
              calib_all[key] = np.array([float(x) for x in value.split()])

      # reshape matrices
      calib_out = {}
      calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
      h, w, _ = self.img.shape 

      # fixed fx fy
      calib_out['P2'][0][0] *= self.proj_W/w
      calib_out['P2'][1][1] *= self.proj_H/h

      # fixed cx cy
      calib_out['P2'][0][2] *= self.proj_W/w
      calib_out['P2'][1][2] *= self.proj_H/h

      calib_out['Tr'] = np.identity(4)  # 4x4 matrix
      calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

      return calib_out
  
  def open_scan(self, filename, calib=None, img=None, theta=None):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    
    # set calib 
    if calib is not None:
      self.img = img
      calib = self.read_calib(calib)
      proj_matrix = calib['P2']@calib['Tr']
      self.proj_matrix = proj_matrix.astype(np.float32)
    
    if theta is not None:
      self.theta = theta

    return self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      return self.do_range_projection()

  def select_points_in_frustum(self, points_2d, x1, y1, x2, y2):
      """
      Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
      :param points_2d: point cloud projected into 2D
      :param points_3d: point cloud
      :param x1: left bound
      :param y1: upper bound
      :param x2: right bound
      :param y2: lower bound
      :return: points (2D and 3D) that are in the frustum
      """
      keep_ind = (points_2d[:, 0] > x1) * (points_2d[:, 1] > y1) * \
                  (points_2d[:, 0] < x2) * (points_2d[:, 1] < y2)

      return keep_ind
  
  def rotation_matrix(self, theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    rz = np.array([cos, -sin, 0, sin, cos, 0, 0, 0, 1]).reshape(3,3)    
    rotmat = np.identity(4)
    rotmat[:3, :3] = rz
    return rotmat
  
  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    if self.front is True:
      keep_idx = self.points[:,0] > 0
      points_hcoords = np.concatenate([self.points[keep_idx], np.ones([keep_idx.sum(),1], dtype=np.float32)], axis=1) # x y z 1
      img_points = (self.proj_matrix @ points_hcoords.T).T
      img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points # x y / z -> return x y 
      keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, self.proj_W, self.proj_H) 
      keep_idx[keep_idx] = keep_idx_img_pts
      self.keep_idx = keep_idx
      img_points = np.fliplr(img_points)

      points_img = img_points[keep_idx_img_pts] # proj x, proj y 
      proj_y, proj_x = points_img[:,0], points_img[:,1]
      depth = np.linalg.norm(self.points, 2, axis=1) # 12466(N)
      depth = depth[keep_idx]

    elif self.front == 360:
      points_hcoords = np.concatenate([self.points, np.ones([self.points.shape[0], 1], dtype=np.float32)], axis=1)
      img_points = (self.proj_matrix @ self.rotation_matrix(self.theta) @ points_hcoords.T).T
      img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], 1)
      self.keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, self.proj_W, self.proj_H) # 123389 -> 40964
      if np.degrees(self.theta) <= 50:
        self.keep_idx_img_pts = self.keep_idx_img_pts & (self.points[:, 0] > 0) # 40964 -> 19960
      elif np.degrees(self.theta) <= 130 :
        self.keep_idx_img_pts = self.keep_idx_img_pts & (self.points[:, 1] < 0)
      elif np.degrees(self.theta) <= 230:
        self.keep_idx_img_pts = self.keep_idx_img_pts & (self.points[:, 0] < 0)
      elif np.degrees(self.theta) <= 310:
        self.keep_idx_img_pts = self.keep_idx_img_pts & (self.points[:, 1] > 0)
      elif np.degrees(self.theta) <= 359:
        self.keep_idx_img_pts = self.keep_idx_img_pts & (self.points[:, 0] > 0) # 40964 -> 19960

      img_points = np.fliplr(img_points)
      img_points = img_points[self.keep_idx_img_pts]
      proj_y, proj_x = img_points[:,0], img_points[:,1]
      depth = np.linalg.norm(self.points[self.keep_idx_img_pts], 2, 1)
      self.keep_idx = None
    
    else:
      # laser parameters
      fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
      fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
      fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

      # get depth of all points
      depth = np.linalg.norm(self.points, 2, axis=1)

      # get scan components
      scan_x = self.points[:, 0]
      scan_y = self.points[:, 1]
      scan_z = self.points[:, 2]

      # get angles of all points
      yaw = -np.arctan2(scan_y, scan_x)
      pitch = np.arcsin(scan_z / (depth + 1e-8))

      # get projections in image coords
      proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
      proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

      # scale to image size using angular resolution
      proj_x *= self.proj_W                              # in [0.0, W]
      proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # # order in decreasing depth
    # depth = np.where(depth>60, 0, depth)
    # depth = (depth-depth.min())/(depth.max()-depth.min())

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    self.order = order

    # depth = np.where(depth>60, 0, depth)
    # depth = (depth-depth.min())/(depth.max()-depth.min())

    # assing to images
    self.proj_range[proj_y, proj_x] = depth*255
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission*255
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = ((self.proj_idx > 0).astype(np.float32))*255

    # cv2.imwrite('test.png', self.proj_range[self.crop_size:]*10)
    
    if self.crop_size is not None:
      return {'range':self.proj_range, 
              'xyz':self.proj_xyz,
              'remission':self.proj_remission,
              'idx':self.proj_idx,
              'mask':self.proj_mask,
              'unproj_range':self.unproj_range[order],
              # 'px':self.points[:,0], 'py':self.points[:,1],
              # 'x':self.points[:,0][order], 'y':self.points[:,1][order]
              'px':proj_x, 'py':proj_y
              # 'px':self.keep_idx, 'py':self.keep_idx
              }
    else:        
      return {'range':self.proj_range, 
              'xyz':self.proj_xyz,
              'remission':self.proj_remission,
              'idx':self.proj_idx,
              'mask':self.proj_mask
              }


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self, sem_color_dict=None, learning_map=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, front=False, crop_size=None):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()
    import yaml
    CFG = yaml.safe_load(open('/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti.yaml', 'r'))
    sem_color_dict = CFG['color_map']

    # make semantic colors
    max_sem_key = 0
    for key, data in sem_color_dict.items():
      if key + 1 > max_sem_key:
        max_sem_key = key + 1
    self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
      self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)
    self.front = front
    self.crop_size = crop_size
    self.learning_map = learning_map
    

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=float)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=float)              # [H,W,3] color

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))
    self.label = label

    return self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half
      self.inst_label = label >> 16    # instance id in upper half
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # set it

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())
    if self.front == 360:
      # label = label[self.keep_idx]
      self.sem_label = self.sem_label[self.keep_idx_img_pts]
      self.inst_label = self.inst_label[self.keep_idx_img_pts]
    elif self.front is True:
      self.sem_label = self.sem_label[self.keep_idx]
      self.inst_label = self.inst_label[self.keep_idx]
    if self.project:
      return self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]
    # instances
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
    
    # cv2.imwrite('test.png',self.proj_sem_color[self.crop_size:]*255)
    if self.crop_size:
      return {'label': self.proj_sem_label,
              'label_c':self.proj_sem_color,
              'label_3d':self.sem_label[self.order]}
    else:
        return {'label':self.proj_sem_label}

  def to_original(self, label):
    # put label in original values
    return self.map(label, self.learning_map_inv)
  
  def map(self, label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


if __name__ == "__main__":
  import cv2
  x = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/08/velodyne/000400.bin'
  y = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/08/labels/000400.label'
  img = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/08/image_2/000400.png'
  calib = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences/08/calib.txt'
  cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti2.yaml'

  h, w = 384, 1248*3
  # h, w = 376, 1241
  # h, w = 64, 1024 
  
  front = 360
  if front is not False:
    crop_size = 128
  else:
    crop_size = None
  import yaml
  CFG = yaml.safe_load(open(cfg_path, 'r'))

  sem_color_dict = CFG['color_map']

  scan = LaserScan(project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  scan = SemLaserScan(sem_color_dict=sem_color_dict, project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)


  # scan = LaserScan(project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  # scan = SemLaserScan(sem_color_dict=CFG['color_map'], project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  # from time import sleep
  # from tqdm import tqdm
  
  # thetas = [i for i in range(50)] + [i for i in range(130, 230)] + [i for i in range(310, 360)]
  # if front is True:
  #   img = cv2.imread(img)
  #   scan.open_scan(x, calib, img)
  # elif front == 360:
  #   img = cv2.imread(img)
  #   for theta in tqdm(thetas):
  #     scan.open_scan(x, calib, img, theta=np.radians(theta))
  #     scan.open_label(y)
      
  #     sleep(0.25)
  # else:
  #   scan.open_scan(x)

  # scan.open_label(y)
  
  #####################################
  # import random 
  # thetas = [i for i in range(50)] + [i for i in range(130, 230)] + [i for i in range(310, 360)]
  # theta =  random.choice(thetas)
  # print('\ntheta', theta)
  # img = cv2.imread(img)
  
  # x = scan.open_scan(x, calib, img, theta=np.radians(theta))
  # y = scan.open_label(y)
  #####################################

  # h, w = 384, 1024*4
  # front = False
  # crop_size = 128
  # scan = LaserScan(project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  # scan = SemLaserScan(sem_color_dict=CFG['color_map'], project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  
  # scan.open_scan(x)
  # scan.open_label(y)
  #####################################

  # thetas = [i for i in range(50)] + [i for i in range(130, 230)] + [i for i in range(310, 360)]
  # thetas = [0, 81, 161, 241, 321]
  # img = cv2.imread(img)
  # for theta in thetas:  
  #   x_ = scan.open_scan(x, calib, img, theta=np.radians(theta))
  #   y_ = scan.open_label(y)

  #####################################
  # resize 48x2048
  
  front = True
  h, w = 384, 1248
  crop_size = 128

  scan = LaserScan(project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  scan = SemLaserScan(sem_color_dict=sem_color_dict, project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  img = cv2.imread(img)
  x = scan.open_scan(x, calib, img)
  y = scan.open_label(y)

  #####################################
  # resize 64x2048
  
  # front = False
  # h, w = 64, 1024
  # crop_size = None

  # scan = LaserScan(project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  # scan = SemLaserScan(sem_color_dict=sem_color_dict, project=True, H=h, W=w, fov_up=3.0, fov_down=-25.0, front=front, crop_size=crop_size)
  # img = cv2.imread(img)
  # x = scan.open_scan(x, calib, img)
  # y = scan.open_label(y)

    
