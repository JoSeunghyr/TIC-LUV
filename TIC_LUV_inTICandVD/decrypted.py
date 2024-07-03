import pyAesCrypt

def Encryption(input_file_path, output_file_path, key):
    pyAesCrypt.encryptFile(input_file_path, output_file_path, key)
    print("File has been encrypted")

def Decryption(input_file_path, output_file_path, key):
    pyAesCrypt.decryptFile(input_file_path, output_file_path, key)
    print("File has been decrypted")


filename = "./models/dmuv.py"  # tica.py ./extract_tic/key_tics.py
Decryption(filename, 'F:/01FeikeLungVideo/ClinicalCode_github/TIC_LUV_v.tic.vd/ticluv/models/dmuv_ec.py',
           key='Please send email to the corresponding authors')
