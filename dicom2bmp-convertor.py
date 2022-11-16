from utils import _dicom_convertor

# Diretório em que está localizado as pastas com as imagens DICOM e o de exportação para BMP.
dicom_dir = "C:\\Users\\gbgui\\GitHub\\DICOM-to-BMP\\DICOM_files"
bmp_dir = "C:\\Users\\gbgui\\GitHub\\DICOM-to-BMP\\BMP_files"


# Função do pacote dicom2jpg para converter os arquivos DICOM da pasta dicom_dir para BMP e armazenar na pasta bmp_dir.
def dicom2bmp(origin, target_root=None, anonymous=False, multiprocessing=True):
    return _dicom_convertor(origin, target_root, filetype='bmp', multiprocessing=multiprocessing, anonymous=anonymous)


if __name__ == '__main__':
    dicom2bmp(dicom_dir, bmp_dir)
