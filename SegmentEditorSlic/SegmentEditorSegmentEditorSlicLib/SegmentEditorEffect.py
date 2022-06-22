import logging
import os

import qt
import vtk

import slicer

from SegmentEditorEffects import *


class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
  """This effect uses Watershed algorithm to partition the input volume"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'SegmentEditorSlic'
    scriptedEffect.perSegment = False # this effect operates on all segments at once (not on a single selected segment)
    scriptedEffect.requireSegments = False # this effect requires segment(s) existing in the segmentation
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
    return """Existing segments are grown to fill the image.
The effect is different from the Grow from seeds effect in that smoothness of structures can be defined, which can prevent leakage.
To segment a single object, create a segment and paint inside and create another segment and paint outside on each axis.
"""

  def setupOptionsFrame(self):

     # Object scale slider
    self.objectScaleMmSlider = slicer.qMRMLSliderWidget()
    self.objectScaleMmSlider.setMRMLScene(slicer.mrmlScene)
    self.objectScaleMmSlider.quantity = "length" # get unit, precision, etc. from MRML unit node
    self.objectScaleMmSlider.minimum = 0
    self.objectScaleMmSlider.maximum = 10
    self.objectScaleMmSlider.value = 2.0
    self.objectScaleMmSlider.setToolTip('Increasing this value smooths the segmentation and reduces leaks. This is the sigma used for edge detection.')
    self.scriptedEffect.addLabeledOptionsWidget("Object scale:", self.objectScaleMmSlider)
    self.objectScaleMmSlider.connect('valueChanged(double)', self.updateMRMLFromGUI)

    # Apply button
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.objectName = self.__class__.__name__ + 'Apply'
    self.applyButton.setToolTip("Accept previewed result")
    self.scriptedEffect.addOptionsWidget(self.applyButton)
    self.applyButton.connect('clicked()', self.onApply)

  def createCursor(self, widget):
    # Turn off effect-specific cursor for this effect
    return slicer.util.mainWindow().cursor

  def setMRMLDefaults(self):
    self.scriptedEffect.setParameterDefault("ObjectScaleMm", 2.0)

  def updateGUIFromMRML(self):
    objectScaleMm = self.scriptedEffect.doubleParameter("ObjectScaleMm")
    wasBlocked = self.objectScaleMmSlider.blockSignals(True)
    self.objectScaleMmSlider.value = abs(objectScaleMm)
    self.objectScaleMmSlider.blockSignals(wasBlocked)

  def updateMRMLFromGUI(self):
    self.scriptedEffect.setParameter("ObjectScaleMm", self.objectScaleMmSlider.value)

  def onApply(self):

    # Get segmentation node and potentially existing segments
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    segmentIds = segmentationNode.GetSegmentation().GetSegmentIDs()  # export all segments

    # This can be a long operation - indicate it to the user
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    # Allow users revert to this state by clicking Undo
    self.scriptedEffect.saveStateForUndo()

    # Get master volume from segmentation
    masterVolume = self.scriptedEffect.parameterSetNode().GetMasterVolumeNode()

    #inputVolume = self.scriptedEffect.masterVolumeImageData()
    inputVolume = masterVolume

    if not inputVolume :
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Create a superpixel segmentation with SimpleITK SLIC filter
    # Using an Otsu filter and BinaryFillhole as a background mask
    ## Needed packages installation
    import SimpleITK as sitk

    ##1. Retrieve the inpute volume as a np array
    inputVolumeAsArray = slicer.util.arrayFromVolume(inputVolume)

    #2.Apply SLic algorythm with choosen parameters on the input volume 

    # Create a sitk image
    image = sitk.GetImageFromArray(inputVolumeAsArray)

    # Define slic filter
    slicFilter = sitk.SLICImageFilter()

    slicLabel = slicFilter.Execute(image)

    #3. Background Treatment

    # Generate a background mask with Otsu and FillHole
    otsuMask = sitk.OtsuThreshold(image,0,1) # 0 is the background value
    filledOtsuMask = sitk.BinaryFillhole(otsuMask)

    slicArray = sitk.GetArrayFromImage(slicLabel)
    filledArray = sitk.GetArrayFromImage(filledOtsuMask)

    x,y,z = slicArray.shape

    number_of_segments = slicArray.max # See if any difference between slicArray[i,j,k]=0 or number_of_segments+1
    try :
        slicArray.shape == filledArray.shape
    except : 
        print("Error slic array and otsu mask have different sizes")

    for i in range(0, x-1):
        for j in range(0, y-1):
            for k in range(0, z-1):
                if filledArray[i,j,k] == 0:# if voxel is in background
                    slicArray[i,j,k]=0 #Replace the value in slic segmentation

    # Convert sitk image to array
    slicLabelArray = sitk.GetArrayFromImage(slicLabel)
    print('Input VolumeAsArray', inputVolumeAsArray)
    print('Labels', slicLabel)

    # IJKtoRAS coordinate system
    ijkToRas = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(ijkToRas)
 
    #Create node for labelmap
    labelmapVolumeNode = slicer.util.addVolumeFromArray(slicLabelArray, ijkToRAS=ijkToRas, name='SlicLabels', nodeClassName='vtkMRMLLabelMapVolumeNode')

    # Add segments to segmentation node from labelmap
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)    

    qt.QApplication.restoreOverrideCursor()

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
