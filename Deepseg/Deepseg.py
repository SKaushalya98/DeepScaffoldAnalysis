# import required python packages
import os
import sys
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
slicer.util.pip_install("pip -U")
try:
  import math
except:
  slicer.util.pip_install("math")
  import math
try:
  import numpy as np
except:
  slicer.util.pip_install("numpy~=1.19.2")
  import numpy as np
try:
  import itk 
except:
  slicer.util.pip_install("itk")
  import itk 
try:
  import cv2
except:
  slicer.util.pip_install("opencv-python==4.5.4.60")
  import cv2  
try:
  import keras
  from keras import backend as K
  from keras.models import Model, load_model
except:
  slicer.util.pip_install("keras")
  import keras
  from keras import backend as K
  from keras.models import Model, load_model
try:
  import tensorflow as tf
except:
  slicer.util.pip_install("tensorflow")
  import tensorflow as tf  
try:
  from scipy import ndimage
except:
  slicer.util.pip_install("scipy")
  from scipy import ndimage
try:
  from skimage.transform import resize
  from skimage.exposure  import rescale_intensity
except:
  slicer.util.pip_install("scikit-image")
  from skimage.transform import resize
  from skimage.exposure  import rescale_intensity
try:
  from patchify import patchify, unpatchify
except:
  slicer.util.pip_install("patchify")
  from patchify import patchify, unpatchify  
from PIL import Image

class Deepseg(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Deepseg" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["G.Ganesharaj & S.Kaushalya (University of Moratuwa)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# CNNSegWidget
#
class DeepsegWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  
  def setup(self):

    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Inputs"
    self.layout.addWidget(inputCollapsibleButton)

    outputsCollapsibleButton = ctk.ctkCollapsibleButton()
    outputsCollapsibleButton.text = "Outputs"
    self.layout.addWidget(outputsCollapsibleButton)

    # Layout within the dummy collapsible button
    inputsFormLayout = qt.QFormLayout(inputCollapsibleButton)
    outputsFormLayout = qt.QFormLayout(outputsCollapsibleButton)
    
    #
    # input volume selector 
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick a Volume" )
    inputsFormLayout.addRow("Input Volume: ", self.inputSelector)

    self.outputBoneClusterSelector = slicer.qMRMLNodeComboBox()
    self.outputBoneClusterSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputBoneClusterSelector.selectNodeUponCreation = True
    self.outputBoneClusterSelector.addEnabled = True
    self.outputBoneClusterSelector.removeEnabled = True
    self.outputBoneClusterSelector.noneEnabled = True
    self.outputBoneClusterSelector.showHidden = False
    self.outputBoneClusterSelector.showChildNodeTypes = False
    self.outputBoneClusterSelector.setMRMLScene( slicer.mrmlScene )
    self.outputBoneClusterSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Semented Volume: ", self.outputBoneClusterSelector)
    

    self.enableWeek4Volume = qt.QCheckBox()
    self.enableWeek4Volume.checked = 0
    self.enableWeek4Volume.setToolTip("Select if the input volume is from week 4 explanation time.")
    inputsFormLayout.addRow("Explanation Times: Week 4", self.enableWeek4Volume)
  
    self.enableWeek10Volume = qt.QCheckBox()
    self.enableWeek10Volume.checked = 0
    self.enableWeek10Volume.setToolTip("Select if the input volume is from week 10 explanation time.")
    inputsFormLayout.addRow("                              Week 10", self.enableWeek10Volume)
    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    inputsFormLayout.addRow("Enable Screenshots: ", self.enableScreenshotsFlagCheckBox)
  
    
  
    self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    self.segmentEditorWidget.rotateSliceViewsToSegmentation()

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)

    #
    # Advanced Button
    #
    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced"
    self.layout.addWidget(advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)
    

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputBoneClusterSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    

    # Add vertical spacer
    self.layout.addStretch(1)
    
    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  
  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode()  and self.outputBoneClusterSelector.currentNode()

  def onApplyButton(self):
    logic = DeepsegLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    enableWeek4VolumeFlag = self.enableWeek4Volume.checked
    enableWeek10VolumeFlag= self.enableWeek10Volume.checked
  
    logic.run(self.inputSelector.currentNode(), self.outputBoneClusterSelector.currentNode(), enableScreenshotsFlag,enableWeek4VolumeFlag,enableWeek10VolumeFlag)
    
    
#
# CNNSegLogic
#
class DeepsegLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode,outputBoneClusterNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    
    if not outputBoneClusterNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False  
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)
 
     
  def run(self, inputVolume, outputBoneClusterVolume, enableScreenshots=0,enableWeek4Volume=0,enableWeek10Volume=0):
 
 
    """
    Run the actual algorithm
    """
    
    
    if not self.isValidInputOutputData(inputVolume, outputBoneClusterVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False
      
    import time
    startTime = time.time()
    logging.info('Processing started')
    
    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('testTest-Start','MyScreenshot',-1)

    # Load correct segmentation model    
    if enableWeek4Volume:
      week4_model = os.path.join(sys.path[0],'myModel_model1_v3_08_11_v2.h5')
      segmentation_model_week4 = load_model(week4_model)
      segmentation_model = segmentation_model_week4
      (segmentation_model).load_weights(week4_model)
    if enableWeek10Volume:
      week10_model = os.path.join(sys.path[0],'myModel_model2_v3_29_10.h5')
      segmentation_model_week10 = load_model(week10_model)
      segmentation_model = segmentation_model_week10  
      (segmentation_model).load_weights(week10_model)

    ## Load input volumes
    input_image = list(slicer.util.arrayFromVolume(inputVolume))
    input_image = np.asarray(input_image)
    vol_size = inputVolume.GetImageData().GetDimensions()
    vol_size = np.asarray(vol_size)
    image_test  = input_image 
    
    # Display progress bar
    progress = slicer.util.createProgressDialog(parent=None, value=0, maximum=int(image_test.shape[0]))

    # Create patches and do prediction
    segmented_volume = []
    
    for img_i in range(image_test.shape[0]):
      patches = patchify(cv2.cvtColor(image_test[img_i,:,:],cv2.COLOR_GRAY2RGB), (64, 64,3), step=64) 
      predicted_patches = []
      for i in range(patches.shape[0]):
          for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]
            single_patch_input=single_patch         
            single_patch_prediction = segmentation_model.predict(tf.convert_to_tensor(single_patch_input))
            for l in single_patch_prediction:
              single_patch_prediction = tf.argmax(l, axis=-1)
              predicted_patches.append(single_patch_prediction)

      predicted_patches = np.array(predicted_patches)
      predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 1,64,64,1) )
      reconstructed_image = unpatchify(predicted_patches_reshaped, (384,384,1))
      
      im = Image.fromarray((((np.array(reconstructed_image)).reshape(384,384)).astype(np.uint8)))
      segmented_volume.append(np.array(im))
      if progress.wasCanceled:
        break
      progress.labelText = '\nProcessing Slice %s' % (img_i)
      slicer.app.processEvents()
      progress.setValue(img_i)
    
    bone_label = np.array(segmented_volume)
    bone_label = np.where(bone_label==2,255,bone_label)
    bone_label = np.where(bone_label==1,127,bone_label)
    slicer.util.updateVolumeFromArray(outputBoneClusterVolume, bone_label)

      # fix the orientation problem
    outputBoneClusterVolume.SetOrigin(inputVolume.GetOrigin())
    outputBoneClusterVolume.SetSpacing(inputVolume.GetSpacing())
    ijkToRasDirections = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASDirectionMatrix(ijkToRasDirections)
    outputBoneClusterVolume.SetIJKToRASDirectionMatrix(ijkToRasDirections)

      # view the segmentation output in slicer
    slicer.util.setSliceViewerLayers(background=outputBoneClusterVolume)

      # change the tumor color space
    displayNode = outputBoneClusterVolume.GetDisplayNode()
    displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels") #vtkMRMLColorTableNodeRainbow
   
    stopTime = time.time()
    # Close progress bar
    progress.close()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))
   

    return True
    
    
class DeepsegTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_Deepseg1()

  def test_CNNSeg1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = DeepsegLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')	