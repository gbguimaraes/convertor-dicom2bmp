import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import cv2
import numpy as np
from pathlib import Path
import os
import concurrent.futures


def _is_unsupported(ds):
    """
    input ds
    if unsupported, return SOP class name
    if supported, return False
    """
    # to exclude unsupported SOP class by its UID
    # PDF
    if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.104.1':
        return 'Encapsulated PDF Storage'
    # exclude object selection document
    elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.59':
        return 'Key Object Selection Document'
    else:
        return False


def _pixel_process(ds, pixel_array):
    """
    Process the images
    input image info and original pixeal_array
    applying LUTs: Modality LUT -> VOI LUT -> Presentation LUT    
    return processed pixel_array, in 8bit; RGB if color
    """

    # LUTs: Modality LUT -> VOI LUT -> Presentation LUT
    # Modality LUT, Rescale slope, Rescale Intercept
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        # try applying rescale slope/intercept
        # cannot use INT, because rescale slope could be<1 
        rescale_slope = float(ds.RescaleSlope)  # int(ds.RescaleSlope)
        rescale_intercept = float(ds.RescaleIntercept)  # int(ds.RescaleIntercept)
        pixel_array = pixel_array * rescale_slope + rescale_intercept
    else:
        # otherwise, try to apply modality 
        pixel_array = apply_modality_lut(pixel_array, ds)

    # personally prefer sigmoid function than window/level
    # personally prefer LINEAR_EXACT than LINEAR (prone to err if small window/level, such as some MR images)
    if 'VOILUTFunction' in ds and ds.VOILUTFunction == 'SIGMOID':
        pixel_array = apply_voi_lut(pixel_array, ds)
    elif 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter
        window_width = ds.WindowWidth
        # some values may be stored in an array
        if type(window_center) == pydicom.multival.MultiValue:
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)
        if type(window_width) == pydicom.multival.MultiValue:
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)
        pixel_array = _get_LUT_value_LINEAR_EXACT(pixel_array, window_width, window_center)
    else:
        # if there is no window center, window width tag, try applying VOI LUT setting
        pixel_array = apply_voi_lut(pixel_array, ds)

    # Presentation VOI
    # normalize to 8 bit
    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255.0
    # if PhotometricInterpretation == "MONOCHROME1", then inverse; eg. xrays
    if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation == "MONOCHROME1":
        # NOT add minus directly
        pixel_array = np.max(pixel_array) - pixel_array

    # conver float -> 8-bit
    return pixel_array.astype('uint8')


def _get_LUT_value_LINEAR_normalized(data, window, level):
    """
    Adjust according to VOI LUT, window center(level) and width values
    Normalized to 8 bit
    """
    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])
    # C.11.2.1.2 Window Center and Window Width
    # if (x <= c - 0.5 - (w-1) /2), then y = ymin
    # else if (x > c - 0.5 + (w-1) /2), then y = ymax
    # else y = ((x - (c - 0.5)) / (w-1) + 0.5) * (ymax- ymin) + ymin


def _get_LUT_value_LINEAR_EXACT_normalized(data, window, level):
    """
    Adjust according to VOI LUT, window center(level) and width values
    Normalized to 8 bit
    """
    data = np.piecewise(data,
                        [data <= (level - (window) / 2),
                         data > (level + (window) / 2)],
                        [0, 255, lambda data: ((data - level + window / 2) / window * 255)])
    return np.clip(data, a_min=0, a_max=255)


def _get_LUT_value_LINEAR_EXACT(data, window, level):
    """
    Adjust according to VOI LUT, window center(level) and width values
    not normalized
    """
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    data = np.piecewise(data,
                        [data <= (level - (window) / 2),
                         data > (level + (window) / 2)],
                        [data_min, data_max,
                         lambda data: ((data - level + window / 2) / window * data_range) + data_min])
    return data


def _ds_to_file(file_path, target_root, filetype, anonymous=None, patient_dict=None):
    """
    The aim of this function is to help multiprocessing
    return True if OK
    return message for dicom convertor to print out
    """

    # read images and their pixel data
    # if anonymous is True -> precalculate patient_dict -> passed as patient dict 
    ds = pydicom.dcmread(file_path, force=True)

    # to exclude unsupported SOP class by its UID
    is_unsupported = _is_unsupported(ds)
    if is_unsupported:
        rv = f'{file_path} cannot be converted.\n{is_unsupported} is currently not supported'
        return rv

    # load pixel_array 
    # *** This is one of the time-limited step  ***
    pixel_array = ds.pixel_array.astype(float)  # preparing for scaling

    # if pixel_array.shape[2]==3 -> means color files [x,x,3]
    # [o,x,x] means multiframe
    if len(pixel_array.shape) == 3 and pixel_array.shape[2] != 3:
        rv = f'{file_path} cannot be converted.\nMultiframe images are currently not supported'
        # print()
        return rv

    #################
    # Process image #
    #################
    pixel_array = _pixel_process(ds, pixel_array)

    ##########################
    # convert to pixel image #
    ##########################
    # if filetype.lower()=='img':
    #    return pixel_array

    ########################
    # Process to save file #
    ########################
    # Color data already be converted by RGB implicitly by pydicom  
    # !!!!!!!! NOTE: only YBR_RCT, RGB are tested...
    # should be convert to "BGR" due to open-cv's RGB arrangement
    if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation in \
            ['YBR_RCT', 'RGB', 'YBR_ICT', 'YBR_PARTIAL_420', 'YBR_FULL_422', 'YBR_FULL', 'PALETTE COLOR']:
        # pixel_array = cv2.cvtColor(np.float32(pixel_array), cv2.COLOR_RGB2BGR)
        pixel_array[:, :, [0, 2]] = pixel_array[:, :, [2, 0]]

    # get full export file path and file name (anonynmous files are pre-calculated and stored in patient_dict)
    full_export_fp_fn = _get_export_file_path(ds, file_path, target_root, filetype, anonymous, patient_dict)
    # make dir
    Path.mkdir(full_export_fp_fn.parent, exist_ok=True, parents=True)
    # write file
    if filetype == 'jpg':
        image_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # 70, 55
        cv2.imwrite(str(full_export_fp_fn), pixel_array, image_quality)
    else:
        cv2.imwrite(str(full_export_fp_fn), pixel_array)

    return True


def _dicom_convertor(origin, target_root=None, filetype=None, anonymous=False, multiprocessing=True):
    """
    origin: can be a .dcm file or a folder
    target_root: root of output files and folders; default: root of origin file or folder
    filetype: can be jpg, jpeg, png, bmp, or ndarray
    full target file path = target_root/Today/PatientID_filetype/StudyDate_StudyTime_Modality_AccNum/Ser_Img.filetype
    """
    # set file type
    if filetype is None:
        filetype = 'bmp'
    elif filetype.lower() not in ['jpg', 'png', 'jpeg', 'bmp', 'img', 'tiff']:
        raise Exception('Target file type should be jpg, png, or bmp')

    # get root folder (as target_root) and dicom_file_list
    target_root, dicom_file_list = _get_root_get_dicom_file_list(origin, target_root)
    full_path_dict = None

    # process image and export files
    if multiprocessing is True:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            return_future = [executor.submit(_ds_to_file, file_path, target_root, filetype, anonymous, full_path_dict)
                             for file_path in dicom_file_list]
            return_message = [future.result() for future in return_future]
        print("Imagens DICOM convertidas em BMP com sucesso!")
    else:
        return_message = [_ds_to_file(file_path, target_root, filetype, anonymous, full_path_dict)
                          for file_path in dicom_file_list]

    return True


def _get_root_get_dicom_file_list(origin_input, target_root):
    """
    Process origin_input
    if it is a list/tuple -> iterate through all the element
    
    return target root_folder 
    - if specified, return Path object of target_root
    - if unspecified, return parent folder of the first element in origin_input
    return dicom_file_list
    """

    # if it is in list or tuple -> iterate through items
    # origin_list is a list, containing folders and files as Path type
    if isinstance(origin_input, list) or isinstance(origin_input, tuple):
        origin_list = [Path(ori) for ori in origin_input]  # make all items as path
    else:
        origin_list = [Path(origin_input)]
    dicom_file_list = []

    # check file/folder existance and if they are dicom files
    # if file or folder does not exist
    for origin in origin_list:
        if not origin.exists():
            raise OSError(f"File or folder '{origin}' does not exist")
        # if it is a file, then check if it's a dicom and add it to dicom_file_list
        if origin.is_file():
            if origin.suffix.lower() != '.dcm':
                raise Exception('Input file type should be a DICOM file')
            else:
                dicom_file_list.append(origin)
        # if it is a dir, then add all the dicom files into dicom_file_list
        elif origin.is_dir():
            for root, sub_f, file in os.walk(origin):
                for f in file:
                    if f.lower().endswith('.dcm'):
                        file_path_dcm = Path(root) / Path(f)
                        # file_path_exp =  folder_destination / Path(f).with_suffix('.jpg')
                        # stor origin / destination
                        dicom_file_list.append(file_path_dcm)
    # sort the list
    dicom_file_list.sort()

    # set root_folder
    # if target root is not specified, set as same root of origin file 
    if target_root is None:
        if isinstance(origin_input, list) or isinstance(origin_input, tuple):
            # list or tuple -> the first element location
            root_folder = Path(origin_input[0]).parent
        else:
            # single folder or file
            root_folder = Path(origin_input).parent
    else:
        root_folder = Path(target_root)
    return root_folder, dicom_file_list


def _get_export_file_path(ds, file_path, target_root, filetype, anonymous, patient_dict):
    """construct export file path"""

    if anonymous == True:
        # get from pre-calculated dictionary
        full_export_fp_fn = patient_dict[file_path]

    # if no anonymous
    else:
        # Try to get metadata       
        patient_metadata = _get_metadata(ds)
        SeriesNumber = patient_metadata['SeriesNumber']
        InstanceNumber = patient_metadata['InstanceNumber']

        # Full export file path
        full_export_fp_fn = target_root / Path(f"{SeriesNumber}_{InstanceNumber}.{filetype}")

    # print(patient_dict)
    return full_export_fp_fn


def _get_metadata(ds):
    """get patient metadata"""
    metadata = {}
    try:
        metadata['SeriesNumber'] = ds.SeriesNumber  # series number
    except:
        metadata['SeriesNumber'] = 'Ser'
    try:
        metadata['InstanceNumber'] = ds.InstanceNumber
    except:
        metadata['InstanceNumber'] = 'Ins'
    return metadata
