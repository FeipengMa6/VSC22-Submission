import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image
from zipfile import ZipFile


# file = filemap[key]
def load_imgs(file):
    img_list = []
    with ZipFile(file, 'r') as f:
        name_list = f.namelist()
        name_list.sort()
        for file in name_list:
            img_buff = f.read(file)
            img_bytes = BytesIO()
            img_bytes.write(img_buff)
            img = Image.open(img_bytes).convert("RGB")
            #img = cv2.imdecode(np.frombuffer(img_buff, np.uint8), cv2.IMREAD_COLOR)
            img_list.append(np.array(img))
    return img_list

def save_imgs(img_list, file):
    zip = ZipFile(file, 'w')
    for idx, imgs in enumerate(img_list):
        for i, img in enumerate(imgs):
            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
            # _, buffer = cv2.imencode('.jpg', img, encode_param)
            # buffer = bytes(buffer)
            img = Image.fromarray(img) 
            img_bytes = BytesIO()
            img.save(img_bytes, format='jpeg', quality=98)
            buffer = img_bytes.getvalue()
            buffer_id = f"{idx}{str(i).zfill(6)}.jpg"
            zip.writestr(buffer_id, buffer)
    zip.close()






def remove_edges(imgs, img_var, avg_canny):
    # concat_imgs = np.stack(imgs)
    # img_var = concat_imgs.var(axis=0).sum(-1)
    sum_h = img_var.mean(0)
    sum_w = img_var.mean(1)
    start_w, start_h = 0, 0
    end_h, end_w = h, w = img_var.shape
    # canny = [(cv2.Canny(img, 50, 400) > 0).astype(float) for img in imgs]
    # avg_canny = sum(canny) / len(canny)
    threshold = min(max(np.quantile(avg_canny, 0.95), 0.2), avg_canny.mean()+ 0.35)
    canny_fea = (avg_canny > threshold).astype(np.float32)
    canny_h = canny_fea.mean(0)
    canny_w = canny_fea.mean(1)
    qh_idx = list(np.where(canny_w > 0.125 + canny_w.mean())[0])
    qh_idx = [x for x in qh_idx if x not in [0, h - 1]]
    mean_c_threshold = 0.0225
    extra_ratio = 0.3
    high_canny_val = 0.65
    for idx in qh_idx:
        extra = round((idx - start_h)*extra_ratio)
        if idx - start_h < 5:
            continue
        sum_v = np.median(sum_w[start_h:idx - extra]) + sum_w[start_h:idx-extra].mean()
        mean_c = canny_w[start_h:idx-extra].mean()
        # print(idx, extra, sum_v, mean_c)
        if (sum_v < 75) and (mean_c < mean_c_threshold):
            start_h = idx + 1
        elif (sum_v < 250) and (mean_c < mean_c_threshold) and (canny_w[idx] > high_canny_val):
            start_h = idx + 1
    qh_idx.reverse()
    for idx in qh_idx:
        if end_h - idx < 5:
            continue
        extra = round((end_h - idx) * extra_ratio)
        sum_v = np.median(sum_w[idx + extra: end_h]) + sum_w[idx + extra: end_h].mean()
        mean_c = canny_w[idx + extra: end_h].mean()
        # print(idx, extra, sum_v, mean_c, canny_w[idx])
        if (sum_v < 75) and (mean_c < mean_c_threshold):
            end_h = idx
        elif (sum_v < 250) and (mean_c < mean_c_threshold) and (canny_w[idx] > high_canny_val):
            end_h = idx
    qw_idx = list(np.where(canny_h > 0.125 + canny_h.mean())[0])
    qw_idx = [x for x in qw_idx if x not in [0, w - 1]]
    for idx in qw_idx:
        if idx - start_w < 5:
            continue
        extra = round((idx - start_w) * extra_ratio)
        # print(np.median(sum_h[start_w:idx - extra]),  sum_h[start_w:idx - extra].mean())
        sum_v = np.median(sum_h[start_w:idx - extra]) + sum_h[start_w:idx - extra].mean()
        mean_c = canny_h[start_w:idx - extra].mean()
        if (sum_v < 75) and (mean_c < mean_c_threshold):
            start_w = idx + 1
        elif (sum_v < 250) and (mean_c < mean_c_threshold) and (canny_h[idx] > high_canny_val):
            start_w = idx + 1
    qw_idx.reverse()
    for idx in qw_idx:
        # extra = max(min(end_w - idx - 20, 15), 0)
        extra = round((end_w - idx) * extra_ratio)
        if end_w - idx < 5:
            continue
        sum_v = np.median(sum_h[idx+extra:end_w]) + sum_h[idx+extra:end_w].mean()
        mean_c = canny_h[idx+extra:end_w].mean()
        if (sum_v < 75) and (mean_c < mean_c_threshold):
            end_w = idx
        elif (sum_v < 250) and (mean_c < mean_c_threshold) and (canny_h[idx] > 0.65):
            end_w = idx
    return [[x[start_h:end_h, start_w:end_w, :] for x in imgs],
            img_var[start_h:end_h, start_w:end_w],
            avg_canny[start_h:end_h, start_w:end_w]]


def split_imgs(imgs, img_var, avg_canny, gap=5, min_size=120):
    # concat_imgs = np.stack(imgs)
    # img_var = concat_imgs.var(axis=0).sum(-1)
    sum_h = img_var.mean(0)
    sum_w = img_var.mean(1)
    h, w = img_var.shape
    is_middle = False
    start = 0
    res_list = []
    half_gap = gap // 2
    for i in range(h-gap):
        if not is_middle and (sum_w[i : i + gap].mean() > 0.1 or i - start > 50):
            is_middle = True
        elif is_middle and sum_w[i : i + gap].mean() < 0.1:
            if i + half_gap - start > min_size:
                res_list.append([
                    [img[start:i + half_gap, :, :] for img in imgs],
                    img_var[start:i + half_gap, :],
                    avg_canny[start:i + half_gap, :]
                ])
            is_middle = False
            start = i + half_gap
    if res_list or start != 0:
        if h - start > min_size:
            res_list.append([
                [img[start:, :] for img in imgs],
                img_var[start:, :],
                avg_canny[start:, :],
            ])
        if res_list:
            return res_list
    is_middle = False
    for i in range(w-gap) or start != 0:
        #print(i, sum_h[i : i + gap].mean(), is_middle)
        if not is_middle and (sum_h[i : i + gap].mean() > 0.1 or i - start > 50):
            is_middle = True
        elif is_middle and sum_h[i : i + gap].mean() < 0.1:
            if i + half_gap - start > min_size:
                res_list.append([
                    [img[:, start:i + half_gap, :] for img in imgs],
                    img_var[:, start:i + half_gap],
                    avg_canny[:, start:i + half_gap]
                ])
            is_middle = False
            start = i + half_gap
    if res_list or start != 0:
        if w - start > min_size:
            res_list.append([
                [img[:, start:, :] for img in imgs],
                img_var[:, start:],
                avg_canny[:, start:]
            ])
        if res_list:
            return res_list
    # canny = [(cv2.Canny(img, 50, 400) > 0).astype(float) for img in imgs]
    # avg_canny = sum(canny) / len(canny)
    threshold = min(max(np.quantile(avg_canny, 0.95), 0.2), avg_canny.mean()+ 0.3)
    canny_fea = (avg_canny > threshold).astype(np.float32)
    canny_h = canny_fea.mean(0)
    canny_w = canny_fea.mean(1)
    h, w = canny_fea.shape
    qh_idx = list(np.where(canny_w > 0.45 + canny_fea.mean())[0])
    qh_idx.reverse()
    wh_idx = list(np.where(canny_h > 0.45  + canny_fea.mean())[0])
    wh_idx.reverse()
    def cut_h(end=h):
        for idx in qh_idx:
            if end - idx > min_size:
                res_list.append([
                    [x[idx:end, :, :] for x in imgs],
                    img_var[idx:end, :],
                    avg_canny[idx:end, :]
                ]
                )
                end = idx
        if res_list:
            if end > min_size:
                res_list.append([
                    [x[:end, :, :] for x in imgs],
                    img_var[:end, :],
                    avg_canny[:end, :]
                ])
    def cut_w(end=w):
        for idx in wh_idx:
            # print(end - idx)
            if end - idx > min_size:
                res_list.append([
                    [x[:, idx:end, :] for x in imgs],
                    img_var[:, idx:end],
                    avg_canny[:, idx:end]
                ])
                end = idx
        if res_list:
            if end > min_size:
                res_list.append([
                    [x[:, :end, :] for x in imgs],
                    img_var[:, :end],
                    avg_canny[:, :end]
                ])
    if w > h:
        cut_w()
        if res_list:
            return res_list
        cut_h()
        if res_list:
            return res_list
    else:
        cut_h()
        if res_list:
            return res_list
        cut_w()
        if res_list:
            return res_list
    return [[imgs, img_var, avg_canny]]


def clean_imgs(imgs, img_var, avg_canny):
    if len(imgs) < 5:
        return [imgs]
    cut_imgs, cut_img_var, cut_avg_canny = remove_edges(imgs, img_var, avg_canny)
    if min(cut_imgs[0].shape[:2]) < 20:
        return [imgs]
    split = split_imgs(cut_imgs, cut_img_var, cut_avg_canny, min_size=80)
    res = []
    if len(split) == 1 and split[0][0][0].shape == cut_imgs[0].shape:
        #print(split[0][0].shape, cut_imgs[0][0].shape)
        res.extend([x[0] for x in split])
    else:
        for imgs, img_var, avg_canny in split:
            res.extend(clean_imgs(imgs, img_var, avg_canny))
    return res

def to_pil_image(img):
    img = Image.fromarray(img)



def image_process(img_list):
    try:
        imgs = [np.array(x) for x in img_list]
        concat_imgs = np.stack(imgs)
        img_var = concat_imgs.var(axis=0).sum(-1)
        num_img = len(imgs)
        canny_imgs = imgs
        if num_img > 20:
            idxs = np.arange(0, num_img, num_img / 20)
            canny_imgs = [canny_imgs[int(np.round(i))] for i in idxs]
        # canny_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in canny_imgs]
        canny = [(cv2.Canny(img, 50, 400) > 0).astype(float) for img in canny_imgs]
        avg_canny = sum(canny) / len(canny)
        clean_list = clean_imgs(imgs, img_var, avg_canny)
        
        if len(clean_list) > 1 or clean_list[0][0].shape != imgs[0].shape:
            out_imgs = []
            for imgs in clean_list:
                out_imgs.extend([Image.fromarray(x) for x in imgs])
            return True, out_imgs
    except Exception as e:
        #print(f'processing failed: {str(e)}') # no log output
        pass 
    return False, [x for x in img_list]