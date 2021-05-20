

class ReconModel(nn.Module):
  def getval(self):
    return self.meanshape,self.idBase,self.exBase,self.meantex,self.texBase
  def __init__(self, face_model, 
              focal=1015, img_size=224, device='cuda:0'):
      super(ReconModel, self).__init__()
      self.facemodel = face_model

      self.focal = focal
      self.img_size = img_size

      self.device = torch.device(device)

      self.renderer = self.get_renderer(self.device)

      self.kp_inds = torch.tensor(self.facemodel['keypoints']-1).squeeze().long()
      
      meanshape = nn.Parameter(torch.from_numpy(self.facemodel['meanshape'],).float(), requires_grad=False)
      self.meanshape=meanshape

      idBase = nn.Parameter(torch.from_numpy(self.facemodel['idBase']).float(), requires_grad=False)
      self.idBase=idBase

      exBase = nn.Parameter(torch.from_numpy(self.facemodel['exBase']).float(), requires_grad=False)
      self.exBase=exBase

      meantex = nn.Parameter(torch.from_numpy(self.facemodel['meantex']).float(), requires_grad=False)
      self.meantex=meantex

      texBase = nn.Parameter(torch.from_numpy(self.facemodel['texBase']).float(), requires_grad=False)
      self.texBase=texBase

      tri = nn.Parameter(torch.from_numpy(self.facemodel['tri']).float(), requires_grad=False)
      self.tri= tri

      point_buf = nn.Parameter(torch.from_numpy(self.facemodel['point_buf']).float(), requires_grad=False)
      self.point_buf=point_buf

  def get_renderer(self, device):
      R, T = look_at_view_transform(10, 0, 0)
      cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=50,
                                      fov=2*np.arctan(self.img_size//2/self.focal)*180./np.pi)

      lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
                            specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

      raster_settings = RasterizationSettings(
          image_size=self.img_size,
          blur_radius=0.0,
          faces_per_pixel=1,
      )
      blend_params = blending.BlendParams(background_color=[0, 0, 0])

      renderer = MeshRenderer(
          rasterizer=MeshRasterizer(
              cameras=cameras,
              raster_settings=raster_settings
          ),
          shader=SoftPhongShader(
              device=device,
              cameras=cameras,
              lights=lights,
              blend_params=blend_params
          )
      )
      return renderer

  def Split_coeff(self, coeff):
      id_coeff = coeff[:, :150]  # identity(shape) coeff of dim 80
      ex_coeff = coeff[:, 150:225]  # expression coeff of dim 64
      tex_coeff = coeff[:, 225:375]  # texture(albedo) coeff of dim 80
      angles = coeff[:, 375:378]  # ruler angles(x,y,z) for rotation of dim 3
      gamma = coeff[:, 378:405]  # lighting coeff for 3 channel SH function of dim 27
      translation = coeff[:, 405:]  # translation coeff of dim 3

      return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation

  def Shape_formation(self, id_coeff, ex_coeff):
      n_b = id_coeff.size(0)

      face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                    torch.einsum('ij,aj->ai', self.exBase, ex_coeff) + self.meanshape

      face_shape = face_shape.view(n_b, -1, 3)

      return face_shape

  def Texture_formation(self, tex_coeff):
      n_b = tex_coeff.size(0)
      face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex

      face_texture = face_texture.view(n_b, -1, 3)
      return face_texture

  def Projection_block(self, face_shape):
      half_image_width = self.img_size // 2
      batchsize = face_shape.shape[0]
      camera_pos = torch.tensor([0.0,0.0,10.0], device=face_shape.device).reshape(1, 1, 3)
      # tensor.reshape(constant([0.0,0.0,10.0]),[1,1,3])
      p_matrix = np.array([self.focal, 0.0, half_image_width, \
                          0.0, self.focal, half_image_width, \
                          0.0, 0.0, 1.0], dtype=np.float32)

      p_matrix = np.tile(p_matrix.reshape(1, 3, 3), [batchsize, 1, 1])
      reverse_z = np.tile(np.reshape(np.array([1.0,0,0,0,1,0,0,0,-1.0], dtype=np.float32),[1,3,3]),
                          [batchsize,1,1])
      
      p_matrix = torch.tensor(p_matrix, device=face_shape.device)
      reverse_z = torch.tensor(reverse_z, device=face_shape.device)
      face_shape = torch.matmul(face_shape,reverse_z) + camera_pos
      aug_projection = torch.matmul(face_shape,p_matrix.permute((0,2,1)))

      face_projection = aug_projection[:,:,:2]/ \
                      torch.reshape(aug_projection[:,:,2],[batchsize,-1,1])
      return face_projection

  
  def Compute_rotation_matrix(self,angles):
      n_b = angles.size(0)
      sinx = torch.sin(angles[:, 0])
      siny = torch.sin(angles[:, 1])
      sinz = torch.sin(angles[:, 2])
      cosx = torch.cos(angles[:, 0])
      cosy = torch.cos(angles[:, 1])
      cosz = torch.cos(angles[:, 2])

      rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

      if angles.is_cuda: rotXYZ = rotXYZ.cuda()

      rotXYZ[0, :, 1, 1] = cosx
      rotXYZ[0, :, 1, 2] = -sinx
      rotXYZ[0, :, 2, 1] = sinx
      rotXYZ[0, :, 2, 2] = cosx
      rotXYZ[1, :, 0, 0] = cosy
      rotXYZ[1, :, 0, 2] = siny
      rotXYZ[1, :, 2, 0] = -siny
      rotXYZ[1, :, 2, 2] = cosy
      rotXYZ[2, :, 0, 0] = cosz
      rotXYZ[2, :, 0, 1] = -sinz
      rotXYZ[2, :, 1, 0] = sinz
      rotXYZ[2, :, 1, 1] = cosz

      rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

      return rotation.permute(0, 2, 1)

  
  def Rigid_transform_block(self,face_shape, rotation, translation):
      face_shape_r = face_shape.bmm(rotation)
      face_shape_t = face_shape_r + translation.view(-1, 1, 3)

      return face_shape_t

  def get_lms(self, face_shape, kp_inds):
      lms = face_shape[:, kp_inds, :]
      return lms
  def Compute_norm(self, face_shape):
    face_id = self.tri.long() - 1
    shape = face_shape
    v1 = shape[:, face_id[:, 0], :]
    v2 = shape[:, face_id[:, 1], :]
    v3 = shape[:, face_id[:, 2], :]
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = e1.cross(e2)
    

  def forward(self, coeff):

      batch_num = coeff.shape[0]
      
      id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

      face_shape = self.Shape_formation(id_coeff, ex_coeff)
      face_texture = self.Texture_formation(tex_coeff)
      rotation = self.Compute_rotation_matrix(angles)
      face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
      face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
      lms = self.Projection_block(face_lms_t)
      lms = torch.stack([lms[:, :, 0], self.img_size-lms[:, :, 1]], dim=2)
      
      #self.Compute_norm(face_shape) compute norm of each surface
      #rotate norm of face
      #multiply with lighting
      face_color = TexturesVertex(face_texture)

      tri = self.tri - 1
      mesh = Meshes(face_shape_t, tri.repeat(batch_num, 1, 1), face_color)
      rendered_img = self.renderer(mesh)
      rendered_img = torch.clamp(rendered_img, 0, 255)

      return rendered_img, lms, face_texture, mesh





#Intialisinz all the paramters to zero
id_tensor = torch.zeros((1, 150), dtype=torch.float32, requires_grad=True, device='cuda')
tex_tensor = torch.zeros((1, 150), dtype=torch.float32, requires_grad=True, device='cuda')
exp_tensor = torch.zeros((1, 75), dtype=torch.float32, requires_grad=True, device='cuda')
rot_tensor = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True, device='cuda')
gamma_tensor = torch.zeros((1, 27), dtype=torch.float32, requires_grad=True, device='cuda')
trans_tensor = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True, device='cuda')
