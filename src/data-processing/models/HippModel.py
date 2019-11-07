#!/usr/bin/python

#------------------------------------------------------------------------------------------
# Class Model: contains ROI (Left & Right), meta-data vector, and the Label
#------------------------------------------------------------------------------------------

class HippModel:
    # Attributes members
    _hippRight = None # 3D data
    _hippLeft = None
    _hippMetaDataVector = None  # [ID, Date, Class, Age, Sex, MMSE] 
    _hippLabel = None # integer


    # constructor
    def __init__(self, hippLeft, hippRight, hippMetaDataVector, hippLabel):
        self._hippRight = hippRight
        self._hippLeft = hippLeft
        self._hippMetaDataVector = hippMetaDataVector
        self._hippLabel = hippLabel

    # Properties
    @property
    def hippLeft(self):
        return self._hippLeft

    @property
    def hippRight(self):
        return self._hippRight

    @property
    def hippMetaDataVector(self):
        return self._hippMetaDataVector

    @property
    def hippLabel(self):
        return self._hippLabel


    # Setters
    @hippRight.setter
    def hippRight(self, value):
        self._hippRight = value

    @hippLeft.setter
    def hippLeft(self, value):
        self._hippLeft = value

    @hippMetaDataVector.setter
    def _hippMetaDataVector(self, value):
        self._hippMetaDataVector = value
        

    @hippLabel.setter
    def hippLabel(self, value):
        self._hippLabel = value

    # Getters
    @hippRight.getter
    def hippRight(self):
        return self._hippRight

    @hippLeft.getter
    def hippLeft(self):
        return self._hippLeft

    @hippMetaDataVector.getter
    def hippMetaDataVector(self):
        return self._hippMetaDataVector

    @hippLabel.getter
    def hippLabel(self):
        return self._hippLabel


    # deleter
    @hippRight.deleter
    def hippRight(self):
        del self._hippRight

    @hippLeft.deleter
    def hippLeft(self):
        del self._hippLeft

    @hippMetaDataVector.deleter
    def hippMetaDataVector(self):
        del self._hippMetaDataVector

    @hippLabel.deleter
    def hippLabel(self):
        del self._hippLabel

