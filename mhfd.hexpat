// http://www.ipodlinux.org/ITunesDB/#Photo_Database

#pragma endian little

import std.string;

struct MHFD {
    //char     id[4];        // "mhfd"
    u32     headerLen;   // usually 0x84
    u32     totalLen;
    u32     unknown1;
    u32     unknown2;
    u32     numChildren;
    u32     unknown3;
    u32     nextId;
    u64     unknown5;
    u64     unknown6;
    u32     unknown7;
    u32     unknown8;
    u32     unknown9;
    u32     unknown10;
    u32     unknown11;
    padding[headerLen - 0x44];
};

struct MHSD {
    //char     id[4];        // "mhsd"
    u32     headerLen;   // usually 0x60
    u32     totalLen;
    u32     index;        // 1=image list, 2=album list, 3=file list
    padding[headerLen - 0x10];
};

struct MHLI {
    //char     id[4];        // "mhli"
    u32     headerLen;   // usually 0x5c
    u32     numImages;
    padding[headerLen - 0x0c];
};

struct MHII {
    //char  id[4];           // "mhii"
    u32  headerLen;
    u32  totalLen;
    u32  numChildren;
    u32  id;
    u32  songId;
    u32  unknown4;
    u32  rating;
    u32  unknown6;
    u32  originalDate;
    u32  digitizedDate;
    u32  sourceImageSize;
    padding[headerLen - 48];
};

struct MHOD {
    //char  id[4];           // "mhod"
    u32  headerLen;
    u32  totalLen;
    u16  type;
    char unknown;
    char paddingLength;
    
    padding[headerLen - 16];
    
    if (type == 1 || type == 3) {
        u32 stringLength;
        u32 unknown2;
        u32 unknown3;

        if (unknown2 == 1) {
            std::string::NullString content;
        } else if (unknown2 == 2) {
            std::string::NullString16 content;
        }
    }

    //u8    data[total_len - header_len];
};

struct MHNI {
    //char  id[4];           // "mhni"
    u32  headerLen;
    u32  totalLen;
    u32  numChildren;
    u32  corrId;
    u32  ithmbOffset;
    u32  imgSize;
    u16 verticalPadding;
    u16 horizontalPadding;
    u16 imageHeight;
    u16 imageWidth;
    u32  unknown;
    u32 imageSize;
    padding[headerLen - 44];
};

struct MHLA {
    u32 headerLen;
    u32 numChildren;
    padding[headerLen - 12];
};

struct MHBA {
    u32 headerLen;
    u32 totalLen;
    u32 numDataObjChildren;
    u32 numAlbumItemChildren;
    u32 playlistId;
    u32 unknown2;
    u16 unknown3;
    char albumType;
    char playMusic;
    char repeat;
    char random;
    char showTitles;
    char transitionDirection;
    u32 slideDuration;
    u32 transitionDuration;
    u32 unknown7;
    u32 unknown8;
    u64 dbid2;
    u32 prevPlaylistId;
    padding[headerLen - 64];
};

struct MHIA {
    u32 headerLen;
    u32 totalLen;
    u32 unknown1;
    u32 imageId;
    padding[headerLen - 20];
};

struct MHLF {
    u32 headerLen;
    u32 numFiles;
    padding[headerLen - 12];
};

struct MHIF {
    u32 headerLen;
    u32 totalLen;
    u32 unknown1;
    u32 corrId;
    u32 imageSize;
    padding[headerLen - 24];
};

struct Node {
    char magic[4];
    
    if (magic == "mhii") { 
        MHII body;
        
        Node children[body.numChildren];
        
    } else if (magic == "mhod") { 
        MHOD body;
        
        if (body.type == 2) {
            Node child;
        }
    } else if (magic == "mhni") { 
        MHNI body; 
        
        Node children[body.numChildren];
        
    } else if (magic == "mhfd") { 
        MHFD body;
        
        Node children[body.numChildren];
        
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
        
        Node images[body.numImages];
    } else if (magic == "mhla") {
        MHLA body;
        
        Node children[body.numChildren];
    } else if (magic == "mhba") {
        MHBA body;
        
        Node dataObjChildren[body.numDataObjChildren];
        Node albumItemChildren[body.numAlbumItemChildren];
    } else if (magic == "mhia") {
        MHIA body;
    } else if (magic == "mhlf") {
        MHLF body;
        
        Node files[body.numFiles];
    } else if (magic == "mhif") {
        MHIF body;
    }
};

Node mhfd @ 0x0;