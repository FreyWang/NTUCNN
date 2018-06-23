import os
import pickle
import warnings

import skimage
import skimage.color
import skimage.io
import skimage.transform


def main():
    img_root_dir = os.path.join('/home', 'haisheng', 'VOC2012', 'JPEGImages')
    rmpe_root_dir = os.path.join('/home', 'haisheng', 'VOC2012', 'bak', 'bak2',
                                 'rmpe')
    upper_inds = list(range(7, 17))
    lower_inds = list(range(1, 8))
    human_inds = list(range(1, 17))

    with open(
            os.path.join('/home', 'haisheng', 'haisheng', 'voc', 'data',
                         'imdb.pickle'), 'rb') as f:
        imdb = pickle.load(f)

    with open(
            os.path.join('/home', 'haisheng', 'haisheng', 'voc', 'data',
                         'bbdb.pickle'), 'rb') as f:
        bbdb = pickle.load(f)

    pbbdb = {}

    for mode in ('train', 'test', 'eval'):
        for i, sample_name in enumerate(imdb[mode]['names']):
            img_name, pid = sample_name.rsplit('_', 1)
            rmpe_path = os.path.join(rmpe_root_dir, img_name, 'person_' + pid)
            bb = bbdb[sample_name]
            img = skimage.io.imread(
                os.path.join(img_root_dir, img_name + '.jpg'))

            if img.ndim < 3:
                img = skimage.color.gray2rgb(img)

            with open(rmpe_path, 'rb') as f:
                preds, _ = pickle.load(f, encoding='bytes')

            pred = preds[0]
            upper_coords = [pred[e - 1] for e in upper_inds]
            lower_coords = [pred[e - 1] for e in lower_inds]
            human_coords = [pred[e - 1] for e in human_inds]
            upper_bb = gen_bb(upper_coords)
            lower_bb = gen_bb(lower_coords)
            human_bb = gen_bb(human_coords)

            upper_bb = rectified_bb(upper_bb, bb)
            lower_bb = rectified_bb(lower_bb, bb)
            human_bb = rectified_bb(human_bb, bb)

            pbbdb[sample_name] = {}
            pbbdb[sample_name]['upper'] = upper_bb
            pbbdb[sample_name]['lower'] = lower_bb
            pbbdb[sample_name]['human'] = human_bb

            for pos in ('upper', 'lower', 'human'):
                save_dir = os.path.join('/', 'home', 'haisheng', 'VOC2012',
                                        'patches', sample_name)
                save_path = os.path.join(save_dir, pos + '.jpg')

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if os.path.exists(save_path):
                    continue

                x1, y1, x2, y2 = pbbdb[sample_name][pos]
                patch = img[y1:y2, x1:x2, :]

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    patch = skimage.img_as_ubyte(
                        skimage.transform.resize(
                            patch, [224, 224], mode='constant'))

                try:
                    skimage.io.imsave(save_path, patch)
                except:
                    import ipdb
                    ipdb.set_trace()

            print('{:5s}:{:4d}/{:4d}'.format(mode, i,
                                             len(imdb[mode]['names'])))

    with open(
            os.path.join('/', 'home', 'haisheng', 'haisheng', 'voc', 'data',
                         'pbbdb.pickle'), 'wb') as f:
        pickle.dump(pbbdb, f)


def gen_bb(coords, ext_scale=0.35):
    xs = [e[0] for e in coords]
    ys = [e[1] for e in coords]
    x1, x2 = min(xs), max(xs) + 1
    y1, y2 = min(ys), max(ys) + 1
    ext_x = (x2 - x1) * ext_scale
    ext_y = (y2 - y1) * ext_scale
    x1 -= ext_x
    x2 += ext_x
    y1 -= ext_y
    y2 += ext_y

    return x1, y1, x2, y2


def rectified_bb(local_bb, global_bb):
    H, W = global_bb[3] - global_bb[1], global_bb[2] - global_bb[0]
    x_bound = global_bb[0]
    y_bound = global_bb[1]

    x1, y1, x2, y2 = local_bb
    # dirty when no such part is detected
    x1 = min(max(x1, 0), W - 1)
    y1 = min(max(y1, 0), H - 1)
    x2 = max(min(x2, W), x1 + 1)
    y2 = max(min(y2, H), y1 + 1)

    x1 += x_bound
    y1 += y_bound
    x2 += x_bound
    y2 += y_bound

    return int(x1), int(y1), int(x2), int(y2)


if __name__ == '__main__':
    main()
