# CVAT Tools

Generates mask png files from a cvat annotations.xml.

## Usage
You can either use the CLI:
```
CVATTools.exe <input_cvat_xml_file> <output_directory>
```

Or use the class in your own C++ program.

## How it works

For every label a directory is created. In this directory, a mask image will be generated for every label and every image in the annoations.xml.
When a label does not occur in an image, a empty mask will be generated. Also, when a label occurs in an image multiple times, all labels gets merged into one single mask.

Example tree given the CVAT [exmaple.xml](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/).
```
|-car
|---filename000.png
|-plate
|---filename000.png
|-traffic_line
|---filename000.png
|-wheel
|---filename000.png
```

## Build

Requires:
- OpenCV (4+)

Uses:
- [pugixml](https://pugixml.org/)
- [cli11](https://github.com/CLIUtils/CLI11)

# License

[MIT](./License)

