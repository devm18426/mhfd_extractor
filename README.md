# MHFD Extractor

Research into and attempt to extract old iPod data, specifically photos stored in the MHFD (iTunes DB) format.

The relevant sections of the [documentation](http://www.ipodlinux.org/ITunesDB/#Photo_Database) on ipodlinux.org was used extensively.

`mhfd.hexpat` can be loaded with ImHex to parse "Photo Database" files.

`mhfd_parser.py` contains a Python implementation of the parser.

`photo_extractor.py` performs the actual extraction.

The photo database may contain more than one version of each photo, this script will only extract the highest resolution version (480x720). The photos are converted from UYVY to RGB, cropped to their actual size, and saved as PNG.

Using the `-a` flag, all image data (even smaller versions) may be extracted in their raw formats for further external processing.

```bash
# python photo_extractor.py -h
usage: photo_extractor.py [-h] [-t THREADS] [-a] photos_dir output_dir

Extract images from iPod photo database.

positional arguments:
  photos_dir            Path to the directory containing 'Photo Database' file and 'Thumbs' directory.
  output_dir            Path to output directory. PNGs will be written here.

options:
  -h, --help            show this help message and exit
  -t, --threads THREADS
                        Number of threads to use for parallel extraction.
  -a, --all             Extract all image data and store in raw formats.
```

The `photos_dir` argument should point to a directory structured similar to this:

```bash
# ls -la Photos/
total 1148
drwxrwxrwx 1 root root    4096 Jan 1 23:09  .
drwxrwxrwx 1 root root    4096 Jan 1 00:01  ..
-rwxrwxrwx 1 root root 1175392 Jan 1 14:46 'Photo Database'
drwxrwxrwx 1 root root    4096 Jan 1 23:10  Thumbs

# ls -la Photos/Thumbs/
total 1049724
drwxrwxrwx 1 root root      4096 Jan 1 23:10 .
drwxrwxrwx 1 root root      4096 Jan 1 23:09 ..
-rwxrwxrwx 1 root root  28211040 Sep 10  2007 F1015_1.ithmb
-rwxrwxrwx 1 root root 524620800 Sep 10  2007 F1019_1.ithmb
-rwxrwxrwx 1 root root 327628800 Sep 10  2007 F1019_2.ithmb
-rwxrwxrwx 1 root root 189388800 Sep 10  2007 F1024_1.ithmb
-rwxrwxrwx 1 root root   5055300 Sep 10  2007 F1036_1.ithmb
```

Cool examples of failed extraction attempts (interlacing and RGB conversion wasn't working quite right):
<img width="704" height="480" alt="Fail image 1" src="https://github.com/user-attachments/assets/51429571-b1b1-439e-832a-daa1fc8c5a7c" />
<img width="704" height="480" alt="Fail image 2" src="https://github.com/user-attachments/assets/a6a14f9f-f2b6-484d-9f7b-414aa0cc2d8e" />


