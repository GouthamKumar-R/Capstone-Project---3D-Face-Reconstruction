print('start non rigid fitting')
rigid_optimizer = torch.optim.Adam([rot_tensor,trans_tensor,tex_tensor,exp_tensor,id_tensor], lr=0.01)
for i in range(300):
  rigid_optimizer.zero_grad()
  coeff = torch.cat([id_tensor, exp_tensor,
          tex_tensor, rot_tensor,
          gamma_tensor, trans_tensor], dim=1)
  rendered_img, pred_lms,face_texture, _ = model(coeff)
  mask = rendered_img[:, :, :, 3].detach()
  if i>=0:
    print(i)
    img = rendered_img.cpu().squeeze()
    pic = img[:, :, :3].detach().numpy().astype(np.uint8)
    pic2=pic[:,:,::-1]
    cv2.imwrite(f"/content/3DMM-Face-Reconstruction/output/{i}.jpg",pic2)
  photo_loss_val = photo_loss(rendered_img[:, :, :, :3], img_tensor, mask>0)
  lm_loss_val = lm_loss(pred_lms, lms, img_size=TAR_SIZE)
  reg_loss_val = reg_loss(id_tensor, exp_tensor, tex_tensor)
  loss=(photo_loss_val*2)+(lm_loss_val*100)+(reg_loss_val*1e-3)
  if i==0:
    print ("Landmark loss: ", lm_loss_val, " Regression Loss: ", reg_loss_val)
  loss.backward()
  rigid_optimizer.step()
print("Completed Rigid fitting")