// http://www.ipodlinux.org/ITunesDB/#Photo_Database

#pragma endian little

struct MHFD {
    //char     id[4];        // "mhfd"
    u32     header_len;   // usually 0x84
    u32     total_len;
    u32     unknown1;
    u32     unknown2;
    u32     num_children;
    u32     unknown3;
    u32     next_id;
    u64     unknown5;
    u64     unknown6;
    u32     unknown7;
    u32     unknown8;
    u32     unknown9;
    u32     unknown10;
    u32     unknown11;
    padding[header_len - 0x44];
};

struct MHSD {
    //char     id[4];        // "mhsd"
    u32     header_len;   // usually 0x60
    u32     total_len;
    u32     index;        // 1=image list, 2=album list, 3=file list
    padding[header_len - 0x10];
};

struct MHLI {
    //char     id[4];        // "mhli"
    u32     header_len;   // usually 0x5c
    u32     num_images;
    padding[header_len - 0x0c];
};

struct MHII {
    //char  id[4];           // "mhii"
    u32  header_len;
    u32  total_len;
    u32  num_children;
    u32  id;
    u32  song_id;
    u32  unknown4;
    u32  rating;
    u32  unknown6;
    u32  originalDate;
    u32  digitizedDate;
    u32  sourceImageSize;
    padding[header_len - 48];
};

struct MHOD {
    //char  id[4];           // "mhod"
    u32  header_len;
    u32  total_len;
    u16  type;
    char unknown;
    char paddingLength;
    
    padding[header_len - 16];
    
    if (type == 3) {
        u32 stringLength;
        u32 unknown2;
        u32 unknown3;

        char content[stringLength];
        
        padding[total_len - 4 - header_len - stringLength - 8];
    }

    //u8    data[total_len - header_len];
};

struct MHNI {
    //char  id[4];           // "mhni"
    u32  header_len;
    u32  total_len;
    u32  num_children;
    u32  corr_id;
    u32  ithmb_offset;
    u32  img_size;
    u16 verticalPadding;
    u16 horizontalPadding;
    u16 imageHeight;
    u16 imageWidth;
    u32  unknown;
    u32 imageSize;
    padding[header_len - 44];
};

struct Node {
    char magic[4];
    
    if (magic == "mhii") { 
        MHII body;
        
        Node children[body.num_children];
        
    } else if (magic == "mhod") { 
        MHOD body;
        
        if (body.type == 2) {
            Node child;
        }
    } else if (magic == "mhni") { 
        MHNI body; 
        
        Node children[body.num_children];
        
    } else if (magic == "mhfd") { 
        MHFD body;
        
        Node children[body.num_children];
        
    } else if (magic == "mhsd") { 
        MHSD body;
        
        if (body.index == 1) {
            Node imageList;
        } else if (body.index == 2) {
            Node albumList;
        } else if (body.index == 3) {
            Node fileList;
        }
    } else if (magic == "mhli") {
        MHLI body;
        
        Node images[body.num_images];
    }
};

Node mhfd @ 0x0;