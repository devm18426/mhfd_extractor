import datetime
import struct


def read_string(f, length):
    return f.read(length).decode('utf-8', errors='replace')


def read_struct(fmt, f):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))


class MHFD:
    def __init__(self, f):
        (
            self.header_len,
            self.total_len,
            self.unknown1,
            self.unknown2,
            self.num_children,
            self.unknown3,
            self.next_id,
            self.unknown5,
            self.unknown6,
            self.unknown7,
            self.unknown8,
            self.unknown9,
            self.unknown10,
            self.unknown11
        ) = read_struct('<IIIIIIIQQIIIII', f)

        padding_len = self.header_len - 0x44
        f.read(padding_len)


class MHSD:
    def __init__(self, f):
        (
            self.header_len,
            self.total_len,
            self.index
        ) = read_struct('<III', f)

        padding_len = self.header_len - 0x10
        f.read(padding_len)


class MHLI:
    def __init__(self, f):
        (
            self.header_len,
            self.num_images
        ) = read_struct('<II', f)

        f.read(self.header_len - 0x0C)


class MHII:
    def __init__(self, f):
        (
            self.header_len,
            self.total_len,
            self.num_children,
            self.id,
            self.song_id,
            self.unknown4,
            self.rating,
            self.unknown6,
            self.original_date,
            self.digitized_date_seconds,
            self.source_image_size
        ) = read_struct('<IIIIIIIIIII', f)

        f.read(self.header_len - 48)

    @property
    def digitized_date(self):
        hfs_epoch = datetime.datetime(1904, 1, 1)
        return hfs_epoch + datetime.timedelta(seconds=self.digitized_date_seconds)


def read_null_terminated_utf16_string(f, max_length=None):
    # Read bytes until null terminator or max length
    chars = []
    while True:
        # Read 2 bytes (UTF-16 character)
        char_bytes = f.read(2)
        if len(char_bytes) < 2:
            break  # End of file
        char = struct.unpack('<H', char_bytes)[0]  # Unpack the 16-bit character
        if char == 0:  # Null terminator
            break
        chars.append(chr(char))
        if max_length and len(chars) >= max_length:
            break
    return ''.join(chars)


class MHOD:
    def __init__(self, f):
        # Read the fixed-length header fields
        (
            self.header_len,
            self.total_len,
            self.type,
            unknown,
            self.padding_length
        ) = read_struct('<IIHBB', f)

        # Skip remaining padding in header
        f.read(self.header_len - 16)

        self.content = None

        if self.type in (1, 3):
            # Read additional fields for type 3 (string-based content)
            (
                self.string_length,
                self.unknown2,  # Assumed to be encoding
                self.unknown3
            ) = read_struct('<III', f)

            if self.unknown2 == 1:
                # Read the UTF-8 null-terminated string
                content_bytes = bytearray()
                while True:
                    byte = f.read(1)
                    if byte == b'\x00':
                        break
                    content_bytes.extend(byte)

                # Decode the collected bytes to a UTF-8 string
                self.content = content_bytes.decode('utf-8')

            elif self.unknown2 == 2:
                # Read the UTF-16 null-terminated string content
                self.content = read_null_terminated_utf16_string(f, self.string_length)


class MHNI:
    def __init__(self, f):
        (
            self.header_len,
            self.total_len,
            self.num_children,
            self.corr_id,
            self.ithmb_offset,
            self.img_size,
            self.vertical_padding,
            self.horizontal_padding,
            self.image_height,
            self.image_width,
            self.unknown,
            self.image_size
        ) = read_struct('<IIIIIIHHHHII', f)

        f.read(self.header_len - 44)


class MHLA:
    def __init__(self, f):
        (
            self.header_len,
            self.num_children
        ) = read_struct('<II', f)

        f.read(self.header_len - 12)


class MHBA:
    def __init__(self, f):
        (
            self.header_len,
            self.total_len,
            self.num_data_obj_children,
            self.num_album_item_children,
            self.playlist_id,
            self.unknown2,
            self.unknown3,
            self.album_type,
            self.play_music,
            self.repeat,
            self.random,
            self.show_titles,
            self.transition_direction,
            self.slide_duration,
            self.transition_duration,
            self.unknown7,
            self.unknown8,
            self.dbid2,
            self.prev_playlist_id,
        ) = read_struct('<IIIIIIHBBBBBBIIIIQI', f)

        f.read(self.header_len - 64)


class MHIA:
    def __init__(self, f):
        (
            self.header_len,
            self.total_len,
            unknown1,
            self.image_id,
        ) = read_struct('<IIII', f)

        f.read(self.header_len - 20)


class MHLF:
    def __init__(self, f):
        (
            self.header_len,
            self.num_files
        ) = read_struct('<II', f)

        f.read(self.header_len - 12)


class MHIF:
    def __init__(self, f):
        (
            self.header_len,
            self.total_len,
            unknown1,
            self.corr_id,
            self.image_size,
        ) = read_struct('<IIIII', f)

        f.read(self.header_len - 24)


class Node:
    def __repr__(self):
        if hasattr(self, "body"):
            return f"<{self.body.__class__.__name__} Node>"

        return super().__repr__()

    def __init__(self, f):
        self.offset = f.tell()
        self.magic = f.read(4).decode('ascii')
        self.children = []

        self.data_obj_children = None
        self.album_item_children = None
        self.files = None

        if self.magic == "mhfd":
            self.body = MHFD(f)
            for _ in range(self.body.num_children):
                self.children.append(Node(f))

        elif self.magic == "mhsd":
            self.body = MHSD(f)
            if self.body.index == 1:
                self.children.append(Node(f))
            elif self.body.index == 2:
                self.children.append(Node(f))
            elif self.body.index == 3:
                self.children.append(Node(f))

        elif self.magic == "mhli":
            self.body = MHLI(f)
            for _ in range(self.body.num_images):
                self.children.append(Node(f))

        elif self.magic == "mhii":
            self.body = MHII(f)
            for _ in range(self.body.num_children):
                self.children.append(Node(f))

        elif self.magic == "mhni":
            self.body = MHNI(f)
            for _ in range(self.body.num_children):
                self.children.append(Node(f))

        elif self.magic == "mhod":
            self.body = MHOD(f)
            if self.body.type == 2:
                self.children.append(Node(f))

        elif self.magic == "mhla":
            self.body = MHLA(f)
            for _ in range(self.body.num_children):
                self.children.append(Node(f))

        elif self.magic == "mhba":
            self.body = MHBA(f)

            self.data_obj_children = []
            for _ in range(self.body.num_data_obj_children):
                self.data_obj_children.append(Node(f))

            self.album_item_children = []
            for _ in range(self.body.num_album_item_children):
                self.album_item_children.append(Node(f))

        elif self.magic == "mhia":
            self.body = MHIA(f)

        elif self.magic == "mhlf":
            self.body = MHLF(f)

            self.files = []
            for _ in range(self.body.num_files):
                self.files.append(Node(f))

        elif self.magic == "mhif":
            self.body = MHIF(f)

        else:
            print(f"Unknown magic: {self.magic} at offset {self.offset}")


def parse_photo_database(file_path):
    with open(file_path, 'rb') as f:
        return Node(f)
