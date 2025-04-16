import immatch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
from immatch.utils.geometry import warp_kpts
from immatch.datasets.extredataset import ExtreDataBuilder
from immatch.localize.localize import QueryLocalizer

if __name__ == "__main__":
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
<<<<<<< HEAD:Extrebenchmark_test.py
model_name = "eloftr"
=======
model_name = "sp_lightglue"
>>>>>>> f51db74 (add localization_metric):Extrebenchark_test.py

# Initialize model
with open(f"configs/{model_name}.yml", "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)["example"]
model = immatch.__dict__[args["class"]](args)
def matcher(im1, im2): return model.match_pairs(im1, im2)

save_dir = f"/media/guan/ZX1/ExeBenchmark/results/{model_name}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pck_path = f"/media/guan/ZX1/ExeBenchmark/results/{model_name}/pck/"
if not os.path.exists(pck_path):
    os.makedirs(pck_path)
loc_path = f"/media/guan/ZX1/ExeBenchmark/results/{model_name}/loc/"
if not os.path.exists(loc_path):
    os.makedirs(loc_path)


# 数据加载
<<<<<<< HEAD:Extrebenchmark_test.py
ExtreData = ExtreDataBuilder(data_root="./benchmark/class_0")
name = "class_0"
save_path = "temp/extre/" + name + "_" + model_name + ".txt"
ExtreData_test = ExtreData.build_scenes()
=======
for i in tqdm(range(32)):
    ExtreData = ExtreDataBuilder(data_root=f"/media/guan/ZX1/ExeBenchmark/Benchmark/class_{i}/")
    name = f'class_{i}'
    save_path = pck_path + name + '_' + model_name + '.txt'
    loc_txt_path = loc_path + name + '_' + model_name + '.txt'
    ExtreData_test = ExtreData.build_scenes()
>>>>>>> f51db74 (add localization_metric):Extrebenchark_test.py

    for one_test in ExtreData_test:
        val_loader = DataLoader(one_test, num_workers=0,
                                batch_size=1, shuffle=False, pin_memory=True)
        pck_1_tot = 0.0
        pck_3_tot = 0.0
        pck_5_tot = 0.0

<<<<<<< HEAD:Extrebenchmark_test.py
    for batch_idx, batch in enumerate(val_loader):
        im1, im2 = batch["im_A_path"][0], batch["im_B_path"][0]
        depth_ref1, depth_ref2 = batch["depth_A_path"][0], batch["depth_B_path"][0]
        depth1 = cv2.imread(depth_ref1, cv2.IMREAD_UNCHANGED)
        depth1 = torch.tensor(depth1[:, :, 0])[None, ...].to(device)
        depth2 = cv2.imread(depth_ref2, cv2.IMREAD_UNCHANGED)
        depth2 = torch.tensor(depth2[:, :, 0])[None, ...].to(device)

        K1 = batch["K1"].to(device)
        K2 = batch["K2"].to(device)
        T_1to2 = batch["T_1to2"].to(device)
=======
        with open(loc_txt_path, 'w') as f_loc:
            for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                im1, im2 = batch['im_A_path'][0], batch['im_B_path'][0]
                depth_ref1, depth_ref2 = batch['depth_A_path'][0], batch['depth_B_path'][0]
                depth1 = cv2.imread(depth_ref1, cv2.IMREAD_UNCHANGED)
                depth1 = torch.tensor(depth1[:, :, 0])[None, ...].to(device)
                depth2 = cv2.imread(depth_ref2, cv2.IMREAD_UNCHANGED)
                depth2 = torch.tensor(depth2[:, :, 0])[None, ...].to(device)

                K1 = batch['K1'].to(device)
                K2 = batch['K2'].to(device)
                T_1to2 = batch['T_1to2'].to(device)
                # w2c 4*4
                query_pose = batch['query_pose'].to(device)
                reference_pose = batch['reference_pose'].to(device)
>>>>>>> f51db74 (add localization_metric):Extrebenchark_test.py

                matches, _, _, _ = matcher(im1, im2)
                # immatch.utils.plot_matches(im1, im2, matches, radius=2, lines=False, sav_fig=f"matches{batch_idx}.png")
                matches_1to2 = torch.tensor(np.column_stack(
                    (matches[:, 0], matches[:, 1])).reshape(1, -1, 2)).to(device)
                matches_2_hat = torch.tensor(np.column_stack(
                    (matches[:, 2], matches[:, 3])).reshape(1, -1, 2)).to(device)

                # matching eval
                valid, kpts = warp_kpts(
                    matches_1to2.double(),
                    depth1.double(),
                    depth2.double(),
                    T_1to2.double(),
                    K1.double(),
                    K2.double()
                )

                gd = (kpts - matches_2_hat).norm(dim=-1)
                pck_1 = (gd < 1.0).float().mean().nan_to_num()
                pck_3 = (gd < 3.0).float().mean().nan_to_num()
                pck_5 = (gd < 5.0).float().mean().nan_to_num()

                pck_1_tot, pck_3_tot, pck_5_tot = (
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
                # localition eval, match tensor
                qlocalizer = QueryLocalizer(
                    K1=K1.double().squeeze(0),
                    K2=K2.double().squeeze(0),
                    pose2=reference_pose.double().squeeze(0),
                    depth2=depth2.squeeze(0),
                    matches=matches_2_hat.squeeze(0),
                )
                pred_matrix = qlocalizer.main_localize(matches_1to2.squeeze(0), query_pose.squeeze(0).cpu().numpy())

                error_t, error_r = qlocalizer.eval(
                    pred_matrix=pred_matrix,
                    gt_pose=query_pose.squeeze(0),
                )
            
            f_loc.write(str(batch_idx) + ' ' + str(error_t) + ' ' + str(error_r) + '\n')
            

<<<<<<< HEAD:Extrebenchmark_test.py
with open(save_path, "w") as f_w:
    for key in result.keys():
        info = key + " " + str(result[key]) + "\n"
        f_w.write(info)
f_w.close()
=======
    result = {
        "pck_1": pck_1_tot.item() / len(val_loader),
        "pck_3": pck_3_tot.item() / len(val_loader),
        "pck_5": pck_5_tot.item() / len(val_loader),
    }

    with open(save_path, 'w') as f_w:
        for key in result.keys():
            info = key + ' ' + str(result[key]) + '\n'
            f_w.write(info)
    f_w.close()
    f_loc.close()
>>>>>>> f51db74 (add localization_metric):Extrebenchark_test.py
