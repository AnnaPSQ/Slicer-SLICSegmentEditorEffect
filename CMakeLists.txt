cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SegmentEditorSlic)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/AnnaPSQ/Slicer-SLICSegmentEditorEffect")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Anna Pasquier (LINUM)")
set(EXTENSION_DESCRIPTION "Volume segmentation using SLIC algorithm.")
set(EXTENSION_ICONURL "https://github.com/AnnaPSQ/Slicer-SLICSegmentEditorEffect/blob/master/SegmentEditorSlic.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/AnnaPSQ/Slicer-SLICSegmentEditorEffect/blob/master/screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegmentEditorSlic)
add_subdirectory(referencetEditorEffect)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
