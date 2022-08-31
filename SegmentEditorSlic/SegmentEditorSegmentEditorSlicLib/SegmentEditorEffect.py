import logging
import os
import numpy as np

import qt
import vtk

import slicer

import SimpleITK as sitk

from SegmentEditorEffects import *


class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
  """This effect uses Watershed algorithm to partition the input volume"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = "SegmentEditorSlic"
    scriptedEffect.perSegment = False # this effect operates on all segments at once (not on a single selected segment)
    scriptedEffect.requireSegments = False # this effect doesn't require segment(s) existing in the segmentation
    AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

  def clone(self):
    # It should not be necessary to modify this method
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__.replace('\\','/'))
    return clonedEffect

  def icon(self):
    # It should not be necessary to modify this method
    iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.png')
    if os.path.exists(iconPath):
      return qt.QIcon(iconPath)
    return qt.QIcon()

  def helpText(self):
    return """Segments are created using the Simple Linear Iterative Clustering (SLIC) algorithm. 
    Feel free to edit the created segments with other segment editor effects. """

  def setupOptionsFrame(self):

    # Load widget from .ui file. This .ui file can be edited using Qt Designer
    # (Edit / Application Settings / Developer / Qt Designer -> launch).
    uiWidget = slicer.util.loadUI(os.path.join(os.path.dirname(__file__), "SegmentEditorSlic.ui"))
    self.scriptedEffect.addOptionsWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Connections

    self.ui.applySlicKmeansButton.connect('clicked()', self.onApplySlicKmeans)
    self.ui.applySlicButton.connect('clicked()', self.onApplySlic)

  def applySlic(self, imageArray):
    """
    Apply Slic algorithm.

    Args:
      image : NumPy array image

    Returns:
      Numpy Array segmentation array
    """
    image = sitk.GetImageFromArray(imageArray)

    #1. Apply SLIC 
    labels = sitk.SLIC(image, spatialProximityWeight=100.0,
                      enforceConnectivity=True,
                      initializationPerturbation=False)
    return(sitk.GetArrayFromImage(labels))

  def segmentMeanGrayscale3D(self, originalImageArray, segmentationLabelArray):
    """
    Calcul the mean color of each segment of a segmentation.
    For 3D grayscale images.

    Args:
        segmentationLabelArray : NumPy Array segmentation
        originalImageArray : Numpy Array of the original image.

    Returns:
        segmentationMeanColorArray : NumPy Array segmentation which 
                  value of each segment is the meaning of RGB value of the original 
                  image.
        medium_grays : list of mean colors of segments
        segments_arrays : table which save position of each segment 
                          segments_arrays[:,:,:,i] has value 1 for all the position in segment i
    """
    x,y,z = originalImageArray.shape
    segmentationMeanColorArray = np.zeros([x,y,z])
    number_of_segments = segmentationLabelArray.max()+1

    try :
        originalImageArray.shape == segmentationLabelArray.shape
    except : 
        print("Error segmention and original image have different sizes")

    segments_arrays = np.zeros([x, y, z, number_of_segments])
    current_segment = 0
    nombre_pixels_per_segment = np.zeros([number_of_segments])
    medium_grays = np.zeros([number_of_segments])
    # calcul the means value of FGB for each segment
    for i in range(0, x):
      for j in range(0, y):
        for k in range (0, z):
          current_segment = segmentationLabelArray[i,j,k]
          nombre_pixels_per_segment[current_segment] += 1
          medium_grays[current_segment] += originalImageArray[i,j,k]
          segments_arrays[i,j,k,current_segment] += 1

    #Calcul the meaning
    for i in range(number_of_segments):
      if nombre_pixels_per_segment[i] != 0:
        medium_grays[i] = medium_grays[i]/nombre_pixels_per_segment[i]

    return medium_grays, segments_arrays

  def kmeansClassificationGrayscale3D(self, originalImageArray, segmentationLabelArray, clusterNumber):
    """
    Classify segments with K-kmeans algorithm.
    For 3D grayscale images.

    Args:
        originalImageArray : NumPy Array of orriginal volume
        segmentationLabelArray : NumPy Array of segmentation

    Returns:
        pred : Predicted labels for each segment (ndarray, shape=numberof segments)
        segments_arrays : List of 3D Arrays. One Array = one segment. prediction[i]= prediction for segments_arrays[:,:,:,i]
    """
    # Sklearn is needed
    try:
      from sklearn.cluster import KMeans
    except:
      logging.error('You need to import sklearn')
      logging.info("Write the following command line in Python interactor to win time for next utilisation  slicer.util.pip_install('scikit-learn')")
      slicer.util.pip_install('scikit-learn')
      from sklearn.cluster import KMeans

    medium_grays, segments_arrays = self.segmentMeanGrayscale3D(originalImageArray, segmentationLabelArray)
    kmeans = KMeans(n_clusters=clusterNumber, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred = kmeans.fit_predict(medium_grays.reshape(-1,1))

    return pred, segments_arrays

  def backgroundMask(self, volumeImage):
    """
    Generate a background mask with Otsu and FillHole

    Args:
        image: sitk image object from master volume

    Returns:
        backgroundArray : np array representing the background
    """
    otsuMask = sitk.OtsuThreshold(volumeImage,0,1) # 0 is the background value
    filledOtsuMask = sitk.BinaryFillhole(otsuMask)
    
    backgroundArray = sitk.GetArrayFromImage(filledOtsuMask)

    return backgroundArray

  def sliceBySliceBackgroundMask(self, volumeImage):
    """
    Generate a background mask with Otsu and a FillHole filter applied slice by slice
    Pipeline 2 : erode + otsu + file hole slice by slice

    Args:
        image: sitk image object from master volume

    Returns:
        backgroundArray : np array representing the background
    """
    pipeline2 = sitk.GrayscaleErode(volumeImage)
    pipeline2 = sitk.OtsuThreshold(pipeline2,0,1)
    pipeline2 = sitk.BinaryFillhole(pipeline2)
    pipeline2 = sitk.GrayscaleDilate(pipeline2)

    x,y,z = pipeline2.GetSize()
    backgroundArray = np.zeros((z,y,x))
    
    for i in range(z):
        filledOtsuGrayscaleErode2D = sitk.BinaryFillhole(pipeline2[:,:,i])
        slice = sitk.GetArrayFromImage(filledOtsuGrayscaleErode2D)
        backgroundArray[i,:,:]=slice
    
    return backgroundArray

  def removeBackgroundPixelPerPixel(self, slicLabelArray, backgroundArray):
    """
    Remove background from created segmentation pixel per pixel

    Args:
        slicLabelArray : array created by sitk SLIC function
        backgroudArray : array created by one of the 2 background mask methods

    Returns:
        slicLabelArray : slicLabelArray without background
    """
    x,y,z = slicLabelArray.shape
    try:
        slicLabelArray.shape == backgroundArray.shape
    except: 
        print("Error slic array and otsu mask have different sizes")

    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                if backgroundArray[i,j,k] == 0:# if voxel is in background
                    slicLabelArray[i,j,k]=0 #Replace the value in slic segmentation
    return slicLabelArray

  def removeBackgroundSegmentPerSegment(self, slicLabelArray, backgroundArray):
    """
    Remove background from created segmentation segment per segment.
    If 95 percent of the segment pixels are in background mask,
    the segment is considered as part of background and remove.

    Args:
        slicLabelArray : array created by sitk SLIC function
        backgroudArray : array created by one of the 2 background mask methods

    Returns:
        slicLabelArray : slicLabelArray without background
    """
    # Parcours du volume pixel par pixel
    # Observation de la valeur du otsu mask a cette position
    # Si dans background changer la valeur pour 0 (le label slic start a la valeur déterminée : 0)
    x,y,z = slicLabelArray.shape
    number_of_segments = slicLabelArray.max()+1
    print(number_of_segments)
    
    try :
        slicLabelArray.shape == backgroundArray.shape
    except : 
        print("Error slic array and otsu mask have different sizes")

    current_segment = 0
    nombre_pixels_foreground_per_segment = [0]*number_of_segments
    nombre_pixels_per_segment = [0]*number_of_segments

    # Observe the number of background pixels per segment
    for i in range(0, x-1):
        for j in range(0, y-1):
            for k in range(0, z-1):
                current_segment = slicLabelArray[i,j,k]
                nombre_pixels_foreground_per_segment[current_segment] += backgroundArray[i,j,k]
                nombre_pixels_per_segment[current_segment] += 1

    # Give 0 value to segment with 95 percent of background pixels
    for i in range(number_of_segments):
        if nombre_pixels_per_segment[i] > 0 :
            foreground_ratio = nombre_pixels_foreground_per_segment[i]/nombre_pixels_per_segment[i]
            if foreground_ratio < 0.15:
                slicLabelArray = np.where(slicLabelArray == i, 0, slicLabelArray)
    return slicLabelArray

  def onApplySlicKmeans(self):
    """ Create segmentation and classify segments with KMeans algorithm with options chosen by user 
    1. Apply slic on the master volume
    2. Predict segment class with kMeans
    3. Rename segments"""

    # Get segmentation node and potentially existing segments
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()

    # This can be a long operation - indicate it to the user
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    # Allow users revert to this state by clicking Undo
    self.scriptedEffect.saveStateForUndo()
    # Get master volume from segmentation
    masterVolume = self.scriptedEffect.parameterSetNode().GetMasterVolumeNode()
    #inputVolume = self.scriptedEffect.masterVolumeImageData()
    inputVolume = masterVolume
    if not inputVolume :
      raise ValueError("Input volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Create a superpixel segmentation with SimpleITK SLIC filter    
    # Retrieve the input volume as a np array
    inputVolumeAsArray = slicer.util.arrayFromVolume(inputVolume)
    # Create a sitk image
    image = sitk.GetImageFromArray(inputVolumeAsArray)

    #1.Apply SLic algorythm with choosen parameters on the input volume 
    slicLabelArray = self.applySlic(inputVolumeAsArray)

    #2. Predict segment class with KMeans

    clusterNumber = self.ui.clusterNumber.currentIndex + 2 # clusterNumber index start at 0
    prediction, segments_arrays = self.kmeansClassificationGrayscale3D(inputVolumeAsArray, slicLabelArray, clusterNumber)

    # IJKtoRAS coordinate system
    ijkToRas = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(ijkToRas)

    import random
    for i in range(prediction.shape[0]):
      #Create the volume of the segment
      segmentVolumeNode = slicer.util.addVolumeFromArray(segments_arrays[:,:,:,i], ijkToRAS=ijkToRas, name='SlicLabels', nodeClassName='vtkMRMLLabelMapVolumeNode')
      # Add segment to segmentation node from
      slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segmentVolumeNode, segmentationNode)
      # Modify the name of the added segment
      segmentation = segmentationNode.GetSegmentation()
      number_of_segments = segmentation.GetNumberOfSegments()
      current_seg = segmentation.GetNthSegment(number_of_segments-1)
      current_seg.SetName(str(prediction[i]))
      r,g,b = random.random(),  random.random(),  random.random()
      current_seg.SetColor(r,g,b)   

      #Delete LabelMap node   
      slicer.mrmlScene.RemoveNode(segmentVolumeNode)

    #self.scriptedEffect.modifySelectedSegmentByLabelmap(labelmapVolumeNode, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeSet)  
    qt.QApplication.restoreOverrideCursor()
    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

  def onApplySlic(self):
    """ Create segmentation with options chosen by user 
    1. Apply slic on the master volume
    2. Create background mask from master volume according to 
      'Select filters to create background mask' option
    3. Remove the background from created segments with the background_mask and according to 
      'Select method to detach background to segmentation' option"""
    
    # Options selected by user
    
    # 0 : Ostu + File Hole
    # 1 : Otsu + Erode + File Hole slice by slice
    background_mask_index = self.ui.backgroundMask.currentIndex

    # 0 : Pixel per pixel
    # 1 : segment per segment
    remove_background_method_index = self.ui.removeBackgroundMethod.currentIndex

    # Get segmentation node and potentially existing segments
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()

    # This can be a long operation - indicate it to the user
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    # Allow users revert to this state by clicking Undo
    self.scriptedEffect.saveStateForUndo()

    # Get master volume from segmentation
    masterVolume = self.scriptedEffect.parameterSetNode().GetMasterVolumeNode()

    #inputVolume = self.scriptedEffect.masterVolumeImageData()
    inputVolume = masterVolume

    if not inputVolume :
      raise ValueError("Input volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Create a superpixel segmentation with SimpleITK SLIC filter    

    # Retrieve the input volume as a np array
    inputVolumeAsArray = slicer.util.arrayFromVolume(inputVolume)

    # Create a sitk image
    image = sitk.GetImageFromArray(inputVolumeAsArray)

    #1.Apply SLic algorythm with choosen parameters on the input volume 
    slicLabelArray = self.applySlic(inputVolumeAsArray)

    # Background Treatment
    #2 Generate Background Mask
    if background_mask_index == 1: 
      backgroundArray = self.sliceBySliceBackgroundMask(image)
    else:
      backgroundArray = self.backgroundMask(image)

    #3 Remove background from segments
    if remove_background_method_index == 1:
      labelArray = self.removeBackgroundSegmentPerSegment(slicLabelArray, backgroundArray)
    else:
      labelArray = self.removeBackgroundPixelPerPixel(slicLabelArray, backgroundArray)
  
    # IJKtoRAS coordinate system
    ijkToRas = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(ijkToRas)
    print(ijkToRas)
 
    #Create label map node with final segmentation
    labelmapVolumeNode = slicer.util.addVolumeFromArray(labelArray, ijkToRAS=ijkToRas, name='SlicLabels', nodeClassName='vtkMRMLLabelMapVolumeNode')

    # Add segments to segmentation node from labelmap, to be able to modify segments
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode) 

    #Delete LabelMap node   
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode) 

    #self.scriptedEffect.modifySelectedSegmentByLabelmap(labelmapVolumeNode, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeSet)  

    qt.QApplication.restoreOverrideCursor()

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
