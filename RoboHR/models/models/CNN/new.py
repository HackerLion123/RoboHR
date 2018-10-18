import os
import zipfile


def main():
	for file in os.listdir('.')
		with zipfile.ZipFile(file) as zip:
			zip.extractall()

if __name__ == '__main__':
	main()