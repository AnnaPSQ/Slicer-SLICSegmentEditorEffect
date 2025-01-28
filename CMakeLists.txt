cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SegmentEditorSlic)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/AnnaPSQ/Slicer-SLICSegmentEditorEffect#readme")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Anna Pasquier (LINUM, UQAM)")
set(EXTENSION_DESCRIPTION "This extension provides a Segment Editor Effect enabling users to segment volume with the Simple Linear Iterative Clustering (SLIC) algorithm.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/AnnaPSQ/Slicer-SLICSegmentEditorEffect/master/SegmentEditorSlic.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/AnnaPSQ/Slicer-SLICSegmentEditorEffect/master/screenshot.png")
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
