import os
import astra
import numpy as np
import glob
from tqdm import tqdm


def split(n_split, angles, I0, mode='test', geom='parallel'):
    # CT data path
    mode = mode
    data_dir = '{}_data'.format(mode)
    split_dir = '{}_split'.format(mode)
    data_path = os.path.join('data', data_dir, '*.npy')

    # simulated settings
    I0 = I0                    # photon counts
    geom = geom           # geometry, parallel or fanflat
    n = 512                     # image width
    det_width = 1.0             # detector width
    numDec = int(n * 1.5)       # DetectorCount
    n_split = n_split                 # number of split
    angles = angles             # number of angles
    split_angles = int(angles / n_split)
    dso = 1000                  # source-object distance (fanbeam)
    dod = 500                   # object-detector distance (fanbeam)

    # ASTRA Toolbox
    vg = astra.create_vol_geom(n, n)
    if geom == 'parallel':
        pg = astra.create_proj_geom(geom, det_width, numDec, np.linspace(0, np.pi, angles, False))
    elif geom == 'fanflat':
        # ft = np.tan(np.deg2rad(angles / 2))         # compute tan of 1/2 the fan angle
        # det_width = abs(2 * (dso + dod) * ft / numDec)   # width of one detector pixel, calculated based on fan angle
        det_width = 2.0
        pg = astra.create_proj_geom(geom, det_width, numDec,
                                    np.linspace(0, 2*np.pi, angles, False), dso, dod)

    # create directory
    for f in range(n_split):
        dir_name = str(f)
        os.makedirs(os.path.join("data", split_dir, dir_name), exist_ok=True)

    # load clean image
    clean_img = sorted(glob.glob(data_path))
    for file in tqdm(range(len(clean_img))):
        img = np.load(clean_img[file])

        # create sinograms
        proj_id = astra.create_projector('cuda', pg, vg)
        sino_id, sino = astra.create_sino(img, proj_id)
        noisy_sino = astra.add_noise_to_sino(sino, I0)

        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)

        # split sinograms and geometry
        for i in range(n_split):
            sub_sino = noisy_sino[i::n_split, :]
            if geom == 'parallel':
                sub_pg = astra.create_proj_geom(geom, det_width, numDec,
                                                np.linspace(0+i*np.pi/angles, np.pi, split_angles, False))
            elif geom == 'fanflat':
                sub_pg = astra.create_proj_geom(geom, det_width, numDec,
                                                np.linspace(0+i*2*np.pi/angles, 2*np.pi, split_angles, False), dso, dod)

            sub_proj_id = astra.create_projector('cuda', sub_pg, vg)
            sub_sino_id = astra.data2d.create('-sino', sub_pg, sub_sino)
            sub_rec_id = astra.data2d.create('-vol', vg)

            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = sub_rec_id
            cfg['ProjectionDataId'] = sub_sino_id
            cfg['option'] = {'FilterType': 'Ram-Lak'}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            sub_rec = astra.data2d.get(sub_rec_id)

            # save split data
            np.save(os.path.join("data", split_dir, str(i), "rec_" + str(file)), sub_rec)

            # delete cache
            astra.projector.delete(sub_proj_id)
            astra.data2d.delete(sub_sino_id)
            astra.data2d.delete(sub_rec_id)
            astra.algorithm.delete(alg_id)

def FBP(input, angles=1024, I0=1e4, geom='parallel'):
    # simulated settings
    I0 = I0  # photon counts
    geom = geom  # geometry, parallel or fanflat
    n = 512  # image width
    det_width = 1.0  # detector width
    numDec = int(n * 1.5)  # DetectorCount
    angles = angles  # number of angles
    dso = 1000  # source-object distance (fanbeam)
    dod = 500  # object-detector distance (fanbeam)

    vg = astra.create_vol_geom(n, n)
    if geom == 'parallel':
        pg = astra.create_proj_geom(geom, det_width, numDec, np.linspace(0, np.pi, angles, False))
    elif geom == 'fanflat':
        det_width = 2.0
        pg = astra.create_proj_geom(geom, det_width, numDec,
                                    np.linspace(0, 2*np.pi, angles, False), dso, dod)

    # create sinograms
    proj_id = astra.create_projector('cuda', pg, vg)
    _, sino = astra.create_sino(input, proj_id)
    noisy_sino = astra.add_noise_to_sino(sino, I0)
    sino_id = astra.data2d.create('-sino', pg, noisy_sino)

    # reconstruct using FBP
    rec_id = astra.data2d.create('-vol', vg)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {'FilterType': 'Ram-Lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(rec_id)

    return rec



