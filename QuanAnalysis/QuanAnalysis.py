# import required python packages
import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
slicer.util.pip_install("pip -U")
slicer.util.pip_install("dask[array] -U")
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
  slicer.util.pip_install("opencv-python")
  import cv2  
try:
  import SimpleITK as sitk
except:
  slicer.util.pip_install("SimpleITK")
  import SimpleITK as sitk
try:
  from scipy import ndimage
except:
  slicer.util.pip_install("scipy")
  from scipy import ndimage
try:
  from skimage.transform import resize
except:
  slicer.util.pip_install("scikit-image")
  from skimage.transform import resize
try:
  from patchify import patchify, unpatchify
except:
  slicer.util.pip_install("patchify")
  from patchify import patchify, unpatchify  
try:
  import openpnm as op
except:
  slicer.util.pip_install("openpnm==2.6.1")
  import openpnm as op
try:
  import porespy as ps
except:
  slicer.util.pip_install("porespy==2.0.2")
  import porespy as ps
try:
  import wx
except:
  slicer.util.pip_install("matplotlib wxPython")
  import wx
import matplotlib  
matplotlib.use("Agg")
from pylab import *
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology
from skimage import measure
from scipy.stats import norm

class QuanAnalysis(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "QuanAnalysis" # TODO make this more human readable by adding spaces
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
class QuanAnalysisWidget(ScriptedLoadableModuleWidget):
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
    # input volume selector SPECT
    #
    self.inputSelector1 = slicer.qMRMLNodeComboBox()
    self.inputSelector1.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector1.selectNodeUponCreation = True
    self.inputSelector1.addEnabled = False
    self.inputSelector1.removeEnabled = False
    self.inputSelector1.noneEnabled = False
    self.inputSelector1.showHidden = False
    self.inputSelector1.showChildNodeTypes = False
    self.inputSelector1.setMRMLScene( slicer.mrmlScene )
    self.inputSelector1.setToolTip( "Pick Semented Volume" )
    inputsFormLayout.addRow("Segmented Volume: ", self.inputSelector1)

    #
    # input volume selector
    #
    self.inputSelector2 = slicer.qMRMLNodeComboBox()
    self.inputSelector2.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector2.selectNodeUponCreation = True
    self.inputSelector2.addEnabled = False
    self.inputSelector2.removeEnabled = False
    self.inputSelector2.noneEnabled = False
    self.inputSelector2.showHidden = False
    self.inputSelector2.showChildNodeTypes = False
    self.inputSelector2.setMRMLScene( slicer.mrmlScene )
    self.inputSelector2.setToolTip( "Pick Scaffold Mask Volume" )
    inputsFormLayout.addRow("Scaffold Mask: ", self.inputSelector2)

    #
    # output clustering volume selector
    #
    self.outputRegistrationResultSelector = slicer.qMRMLNodeComboBox()
    self.outputRegistrationResultSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputRegistrationResultSelector.selectNodeUponCreation = True
    self.outputRegistrationResultSelector.addEnabled = True
    self.outputRegistrationResultSelector.removeEnabled = True
    self.outputRegistrationResultSelector.noneEnabled = True
    self.outputRegistrationResultSelector.showHidden = False
    self.outputRegistrationResultSelector.showChildNodeTypes = False
    self.outputRegistrationResultSelector.setMRMLScene( slicer.mrmlScene )
    self.outputRegistrationResultSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Registration Result: ", self.outputRegistrationResultSelector)

    #
    # output volume selector
    #
    self.outputVOISelector = slicer.qMRMLNodeComboBox()
    self.outputVOISelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputVOISelector.selectNodeUponCreation = True
    self.outputVOISelector.addEnabled = True
    self.outputVOISelector.removeEnabled = True
    self.outputVOISelector.noneEnabled = "(Overwrite input)"
    self.outputVOISelector.showHidden = False
    self.outputVOISelector.showChildNodeTypes = False
    self.outputVOISelector.setMRMLScene( slicer.mrmlScene )
    self.outputVOISelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Extracted Volume: ", self.outputVOISelector)

    # #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    inputsFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

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
    self.inputSelector1.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputSelector2.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputVOISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    #self.outputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputRegistrationResultSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    
    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()
    

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector1.currentNode() and self.inputSelector2.currentNode() and self.outputRegistrationResultSelector.currentNode() and self.outputVOISelector.currentNode()  

  def onApplyButton(self):
    logic = QuanAnalysisLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    logic.run(self.inputSelector1.currentNode(), self.inputSelector2.currentNode(),self.outputVOISelector.currentNode(), self.outputRegistrationResultSelector.currentNode(), enableScreenshotsFlag)
    scriptFolder = slicer.modules.quananalysis.path.replace('QuanAnalysis.py', '/')
    pm = qt.QPixmap(os.path.join(scriptFolder, "Pore_Throat_size_distribution.png"))
    self.imageWidget = qt.QLabel()
    self.imageWidget.setPixmap(pm)
    self.imageWidget.setScaledContents(True)
    self.imageWidget.show()
    
#
# CNNSegLogic
#
class QuanAnalysisLogic(ScriptedLoadableModuleLogic):
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

  def isValidInputOutputData(self, inputVolumeNode1, inputVolumeNode2 , outputRegistrationResultNode, outputVOINode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode1:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not inputVolumeNode2:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputRegistrationResultNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    
    if not outputVOINode:
      logging.debug('isValidInputOutputData failed: no output cluster node defined')
      return False
    #if inputVolumeNode1.GetID()==outputLesVolumeNode.GetID():
     # logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      #return False
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
  
  # Create fill scaffold mask using the predicted scaffold region
  def create_filled_scaffold(self,img):
    closing_performed=[]
   
    for k in range(img.shape[0]):
      img_i=img[k,:,:]
      thresholdImage=(np.array(img_i==1).astype("uint8"))
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))
      closing = cv2.morphologyEx(thresholdImage, cv2.MORPH_CLOSE, kernel)
      closing=ndimage.binary_fill_holes(closing, structure=None, output=None, origin=0)
    
      closing_performed.append(np.array(self.remove_small_objects(np.array(closing).astype("uint8"))))
    return closing_performed
  # Remove small objects in the morphologically filled scaffold regions in order to increase the registration performance
  def remove_small_objects(self,img):
    binary = img.copy()
    binary[binary>0] = 1
    labels = morphology.label(binary)
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    rank = np.argsort(np.argsort(labels_num))
    index = list(rank).index(len(rank)-2)
    new_img = img.copy()
    new_img[labels!=index] = 0
    return new_img
  
  # Do quantitative analysis in the VOI
  def quantitative_analysis(self,input_image):

    # Scaffold, bone, pore volume ratio calculation
    unique_elements, counts_elements = np.unique(((input_image[:,:,:]>0).astype("uint8")), return_counts=True)
    ROI=counts_elements[1]
    
    unique_elements_scaffold, counts_elements_scaffold = np.unique(((input_image[:,:,:]==200).astype("uint8")), return_counts=True)
    ROI_scaffold=counts_elements_scaffold[1]
    
    unique_elements_bone, counts_elements_bone = np.unique(((input_image[:,:,:]==127).astype("uint8")), return_counts=True)
    ROI_bone=counts_elements_bone[1]
    
    scaffold_volume=float(ROI_scaffold/ROI)
    print(f'Scaffold Volume Ratio: {scaffold_volume}')
    
    bone_volume=float(ROI_bone/ROI)
    print(f'Bone Volume Ratio: {bone_volume}')
    
    pore_volume=1-bone_volume-scaffold_volume
    print(f'Pore Volume Ratio: {pore_volume}')
    
    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(50) 


    volume=np.where(input_image>200,1,0)
    # Calculate pore specific surface area
    se=np.zeros((3,3,3),np.uint8)
    se[1,1,1]=1
    se[1,1,2]=1
    se[1,1,0]=1
    se[1,2,1]=1
    se[1,0,1]=1
    se[2,1,1]=1
    se[0,1,1]=1

    input_image1 = np.where(input_image>200,0,1)
    B=ndimage.binary_dilation(input_image1, structure=se).astype(input_image.dtype)
    C=B-input_image1.astype(input_image.dtype)

    unique_elements_C, counts_elements_C = np.unique(C, return_counts=True)
    pore_uniqueValues, pore_occurCount = np.unique(input_image1 , return_counts=True)
    ROI_C=counts_elements_C[1]
    SP=ROI_C/pore_occurCount[0] # specific surface of pore
    print(f'Pore Specific Surface Area: {SP}')

    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(55) 
    
    
    # Extract network of pore region  and calculate mean pore radius and mean coordination number (connectivity)
    im = ps.tools.align_image_with_openpnm(volume)
    snow = ps.filters.snow_partitioning(im)
    regions = snow.regions*snow.im
    Extracted_network=ps.networks.regions_to_network(regions)
    
    Network=np.zeros((max(Extracted_network["pore.region_label"]),max(Extracted_network["pore.region_label"])))
    FF=Extracted_network["throat.conns"]
    for I in range(0,FF.shape[0]):
      Network[FF[I,0],FF[I,1]] = 1
      if self.progress.wasCanceled:
        break 

    Coordinations=sum(Network)
       
    AVERAGE_COORDINATION_NUMBER=np.mean(Coordinations/Network.shape[0])
    print(f'Average Coordination Number: {AVERAGE_COORDINATION_NUMBER}')
    
    AVERAGE_PORE_RADIUS=np.mean(Extracted_network["pore.equivalent_diameter"]/2)
    print(f'Average Pore Radius: {AVERAGE_PORE_RADIUS}')
    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(90) 
    
    # Plot throat and pore size distribution and save figure
    fig, ax = plt.subplots(2,1)
    ax[0].hist(Extracted_network["throat.equivalent_diameter"]/2,density=True) 
    ax[0].grid(True)
    ax[0].set_title('Throat/Pore size distribution')
    ax[0].set_xlabel('Throat Radious')
    ax[0].set_ylabel('Probability')
    
    ax[1].hist(Extracted_network["pore.equivalent_diameter"]/2,density=True) 
    ax[1].grid(True)
    ax[1].set_xlabel('Pore Radious')
    ax[1].set_ylabel('Probability')
  
    plt.tight_layout()
    scriptFolder = slicer.modules.quananalysis.path.replace('QuanAnalysis.py', '/')
    plt.savefig(os.path.join(scriptFolder, "Pore_Throat_size_distribution.png"))

    # Static image view
    
  

  # Perform 3D image rigid registration
  def multires_registration(self,fixed_image, moving_image, initial_transform):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(f'Final metric value: {registration_method.GetMetricValue()}')
    print(f'Optimizer\'s stopping condition, {registration_method.GetOptimizerStopConditionDescription()}')
    return (final_transform, registration_method.GetMetricValue())	
     
  def run(self, inputVolume1, inputVolume2,outputBoneClusterVolume, outputLesClusterVolume, enableScreenshots=0):
 
 
    """
    Run the actual algorithm
    """
    if not self.isValidInputOutputData(inputVolume1, inputVolume2, outputBoneClusterVolume, outputLesClusterVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False
      
    logging.info('Processing started')

    # Display progress bar
    self.progress = slicer.util.createProgressDialog(parent=None, value=0, maximum=100)
    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('testTest-Start','MyScreenshot',-1)
    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(0)  
    # Load input volume 1

    input_image1 = list(slicer.util.arrayFromVolume(inputVolume1))
    input_image1 = np.asarray(input_image1) # convert volume to nd array

    # Load input volume 2
    input_image2 = list(slicer.util.arrayFromVolume(inputVolume2))
    input_image2 = np.asarray(input_image2)
    
    scaffold=(input_image1[:,:,:]==255)
    fixed_image=self.create_filled_scaffold(scaffold)
   
    all_orientations = {'x=0, y=0, z=90': (0.0,0.0,np.pi/2.0),
                    'x=0, y=90, z=0': (0.0,np.pi/2.0,0.0),
                    'x=0, y=90, z=45': (0.0,np.pi/2.0, np.pi/4.0),
                     'x=0, y=90, z=75': (0.0,np.pi/2.0, np.pi/2.4),
                    'x=0, y=90, z=90': (0.0,np.pi/2.0, np.pi/2.0)}    

  # Registration framework setup. 
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    fixed_image = sitk.GetImageFromArray(np.array(fixed_image))
    moving_image = sitk.GetImageFromArray(np.array(input_image2))   
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
  # Evaluate the similarity metric using the rotation parameter space sampling, translation remains the same for all.
    
    initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed_image,
                      moving_image, 
                      sitk.Euler3DTransform(), 
                      sitk.CenteredTransformInitializerFilter.MOMENTS))
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    best_orientation = (0.0,0.0,0.0)
    best_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
    
    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(10)  
  # Iterate over all other rotation parameter settings. 
    for key, orientation in all_orientations.items():
      initial_transform.SetRotation(*orientation)
      registration_method.SetInitialTransform(initial_transform)
      current_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
      if current_similarity_value < best_similarity_value:
        best_similarity_value = current_similarity_value
        best_orientation = orientation
      if self.progress.wasCanceled:
        break  

    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(25)   

    print('best orientation is: ' + str(best_orientation))
    initial_transform.SetRotation(*best_orientation)
    final_transform,_ = self.multires_registration(fixed_image, moving_image, initial_transform)

    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(40) 

    fixed_image1 = sitk.Cast(sitk.GetImageFromArray(np.array(input_image1)), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image1)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    out = resampler.Execute(moving_image)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed_image1), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    # Registration result
    input_image=sitk.GetArrayFromImage(cimg)
    volume=np.zeros(input_image.shape,dtype="uint8")
    volume=input_image[:,:,:,1]
    volume=np.where(input_image[:,:,:,2]==190,127,volume)
    volume=np.where(input_image[:,:,:,2]==254,200,volume) # VOI

    
    slicer.util.updateVolumeFromArray(outputLesClusterVolume,volume) 
    scriptFolder = slicer.modules.quananalysis.path.replace('QuanAnalysis.py', '/')
    sitk.WriteTransform(final_transform, os.path.join(scriptFolder, "Transform.tfm"))
    # transformFromParent = vtk.vtkTransform()
    # transformFromParent =final_transform
    # outputTransform.SetAndObserveTransformToParent(transformFromParent)

    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(45) 

    # Do quantitative analysis
    self.quantitative_analysis(volume)

    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(95) 

  
    volume1=input_image[:,:,:,1]
    volume1=np.where(input_image[:,:,:,2]==190,127,volume1)
    volume1=np.where(input_image[:,:,:,2]==254,200,volume1)
    volume1=np.where(volume1==255,0,volume1)
    volume1=np.where(volume1==200,255,volume1)

    slicer.util.updateVolumeFromArray(outputBoneClusterVolume, volume1)
    
      # fix the orientation problem
    outputBoneClusterVolume.SetOrigin(inputVolume1.GetOrigin())
    outputBoneClusterVolume.SetSpacing(inputVolume1.GetSpacing())
    ijkToRasDirections = vtk.vtkMatrix4x4()
    inputVolume1.GetIJKToRASDirectionMatrix(ijkToRasDirections)
    outputBoneClusterVolume.SetIJKToRASDirectionMatrix(ijkToRasDirections)

      # view the segmentation output in slicer
    slicer.util.setSliceViewerLayers(background=outputBoneClusterVolume)
    #slicer.util.setSliceViewerLayers(foreground=outputBoneClusterVolume)
    #slicer.util.setSliceViewerLayers(foregroundOpacity=0.5)

      # change the tumor color space
    displayNode = outputBoneClusterVolume.GetDisplayNode()
    displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels") #vtkMRMLColorTableNodeRainbow


    # slicer.util.updateVolumeFromArray(outputBoneClusterVolume,volume1)
    # outputBoneClusterVolume.SetOrigin(inputVolume1.GetOrigin())
    # outputBoneClusterVolume.SetSpacing(inputVolume1.GetSpacing())
    # slicer.util.setSliceViewerLayers(background=outputBoneClusterVolume,fit=True)
        
    self.progress.labelText = '\nProcessing' 
    slicer.app.processEvents()
    self.progress.setValue(100)
    # Close progress bar  
    self.progress.close()
    logging.info('Processing completed')
    


    return True
    
    
class QuanAnalysisTest(ScriptedLoadableModuleTest):
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
    self.test_QuanAnalysis1()

  def test_QuanAnalysis1(self):
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
    logic = QuanAnalysisLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')	