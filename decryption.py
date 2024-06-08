import pyAesCrypt

# key_tics.py, dmuv.py, tica.py have been encrypted
def Decryption(input_file_path, output_file_path, key):
    pyAesCrypt.decryptFile(input_file_path, output_file_path, key)
    print("File has been decrypted")


filename = "."
Decryption(filename, '.', key='.')
