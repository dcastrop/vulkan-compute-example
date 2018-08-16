#ifndef __FILE_CONTENTS__
#define __FILE_CONTENTS__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "utils.h"

typedef struct FileContents {
    uint64_t size;
    const uint8_t *code;
} FileContents_T;
typedef FileContents_T * FileContents;

FileContents readFile(const char* filePath){
    FILE *file;
    uint64_t length;

    file = fopen(filePath, "rb");
    if (!file){
        RUNTIME_ERROR("Cannot open file");
    }

    fseek(file, 0, SEEK_END);
    length = ftell(file);
    rewind(file);

    // Not null terminated
    uint8_t * buffer = (uint8_t *)malloc(length*sizeof(uint8_t));
    fread(buffer, length, 1, file);
    fclose(file);

    FileContents contents =
        (FileContents)malloc(sizeof(FileContents_T));
    contents->size = length;
    contents->code = buffer;

    return contents;
}

void destroyFileContents(FileContents f){
    if(f){
        free((uint8_t *)f->code);
        free(f);
    }
}

#endif