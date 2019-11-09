#!/usr/bin/python

#------------------------------------------------------------------------------------------
# Class Model: contains ROI (Left & Right), meta-data vector, and the Label
#------------------------------------------------------------------------------------------
class Subject:    
    #------------------------------------
    # Attributes members
    #------------------------------------
    _subjectID = ""
    _dateAcqui = ""
    _group = ""
    _age = 0
    _sex = ""
    _mmse = 0

    #------------------------------------
    # constructor
    #------------------------------------
    def __init__(self, subjectID, dateAcqui, group, age, sex, mmse):
        self._subjectID = subjectID
        self._dateAcqui = dateAcqui
        self._group = group
        self._age = age
        self._sex = sex
        self._mmse = mmse

    #------------------------------------
    # Properties
    #------------------------------------
    @property
    def subjectID(self):
        return self._subjectID

    @property
    def dateAcqui(self):
        return self._dateAcqui
    
    @property
    def group(self):
        return self._group
    
    @property
    def age(self):
        return self._age
    
    @property
    def sex(self):
        return self._sex
    
    @property
    def mmse(self):
        return self._mmse
       
    #------------------------------------
    # Access
    #------------------------------------  
    # Setters
    @subjectID.setter
    def subjectID(self, value):
        self._subjectID = value

    @dateAcqui.setter
    def dateAcqui(self, value):
        self._dateAcqui = value

    @group.setter
    def group(self, value):
        self._group = value

    @age.setter
    def age(self, value):
        self._age = value
        
    @sex.setter
    def sex(self, value):
        self._sex = value

    @mmse.setter
    def mmse(self, value):
        self._mmse = value

    # Getters
    @subjectID.getter
    def subjectID(self):
        return self._subjectID

    @dateAcqui.getter
    def dateAcqui(self):
        return self._dateAcqui
    
    @group.getter
    def group(self):
        return self._group

    @age.getter
    def age(self):
        return self._age         

    @sex.getter
    def sex(self):
        return self._sex 

    @mmse.getter
    def mmse(self):
        return self._mmse
        
    # deleter
    @subjectID.deleter
    def subjectID(self):
        del self._subjectID

    @dateAcqui.deleter
    def dateAcqui(self):
        del self._dateAcqui
                
    @group.deleter
    def group(self):
        del self._group

    @age.deleter
    def age(self):
        del self._age         

    @sex.deleter
    def sex(self):
        del self._sex 

    @mmse.deleter
    def mmse(self):
        del self._mmse
    
#------------------------------------------------------------------------
#------------------------------------------------------------------------ 
