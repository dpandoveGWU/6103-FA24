# Object Oriented Programming (OOP) 
# A `Class` is a prototype.
# An `Object` is a live instance of a class.

# class can be thought of bundling properties (like variables) and functions 
# (or called methods) together
# This is not a requirement, but good practice to use Capitalize first letter 
# for classes, variables or functions instances use regular lowercase first letter.

# %%
# We have seen objects before.
mesg = 'Hello, World' # `mesg` is an instance of the class `str`
print( type(mesg) )

# We should be able to use the methods available for the class `str`
# <object><dot><method_name>()
# Ref: https://docs.python.org/3/library/stdtypes.html#str
# Try a method on `mesg`


# Defining our own class
# Syntax: class <ClassName>:
# commented block with class definition

# %%
class Address():
    """
    Defines the class to manipulates US addresses.
    Format: Street, City, State, ZIP
    """

    # constructor and properties
    # __init__ is also called constructor in other programming langs
    # it also set the attributes in here 
    def __init__(self, add):
        items = add.split(',')

        if len(items) != 4:
            raise ValueError('Given address is not properly formatted')
        
        # These following are class attributes
        self.street = items[0].strip()
        self.city = items[1].strip()
        self.state = items[2].strip()
        self.zip = items[3].strip()

    # Class Methods

    # Activity: 1
    # Write a method to check if the address is an apartment
    def isApt(self):
        pass

    
    # Activity: 2
    # Write a method to add apartment info to the address
    def addApt(number):
       pass


    # The print method
    # This is supposed to be an internal method, not to be touched by users
    # def __str__(self):
        # return f"{self.street}, {self.city}, {self.state}, {self.zip}"
    #    pass


# %%
# Create an object from the class
add = Address('221b BakerAPT Street, Washington, D.C., 20201')


# %%
# Check if it is an instance of our class
print( type(add) )
print( isinstance(add, Address) )
print( add )


# %%
# Get and update city
#


# %%
print( )



# %%
# Another Example
class Person:
  """ 
  a person with properties and methods 
  height in meters, weight in kgs
  """

  # constructor and properties
  # __init__ is also called constructor in other programming langs
  # it also set the attributes in here 
  def __init__(self, lastname, firstname, height, weight) :
    self.lastname = lastname
    self.firstname = firstname
    self.height_m = height
    self.weight_kg = weight
  
  # find bmi according to CDC formula bmi = weight/(height^2)
  def bmi(self) : 
    return self.weight_kg/(self.height_m ** 2)
  
  def print_info(self) :
    print( self.firstname, self.lastname+"'s height {0:.{digits}f}m, weight {1:.1f}kg, and bmi currently is {2:.{digits}f}".format(self.height_m, self.weight_kg, self.bmi(), digits=2) )
    return None

  # gain weight
  def gain_weight_kg(self,gain) : 
    self.weight_kg = self.weight_kg + gain 
    # return
    return self

  # gain height
  def gain_height_m(self,gain) : 
    self.height_m = self.height_m + gain 
    # return
    return self
  
  def height_in(self) :
    # convert meters to inches
    return self.height_m *100/2.539
  
  def weight_lb(self) :
    # convert meters to inches
    return self.height_m *100/2.539
  
  
  

#%%
# instantiate the Person object as elo, etc
elo = Person('Lo','Edwin',1.6,60)
vars(elo) # shows all attributes and their values
# dir(elo) # shows all attributes and methods

#%%
elo.print_info()
elo.gain_weight_kg(5) # no return value for this method
# same as
# Person.gain_weight_kg(elo,5) # use both arguments here
elo.print_info()

#%%
superman = Person('Man','Super', 1.99, 85)
superman.gain_weight_kg(-3.5)
superman.print_info()

persons = []
persons.append(elo)
persons.append(superman)
print(len(persons))

#%%
# Add to the Person class four other attributes. At least one of the type float or int.
# Add at least three other methods to the class that might be useful


#%% [markdown]
# 
# ## From a programmer's perspective on Object-Oriented Programming (OOP)
# 
# Read this [blog at Medium on OOP](https://medium.com/@cscalfani/goodbye-object-oriented-programming-a59cda4c0e53). 
# To put all these into context, from procedural progamming (such as C) to OOP (C++, java and the likes) was a 
# huge paradigm shift. The world has progressed however, and there are new needs, and new wants, from the new generations. 
# And there are new answers in response. Keep up with the new ideas and concepts. 
# That's how to stay ahead. 
# Just like OOP still uses a lot of concepts and functionality in procedure programming, 
# the new programming paradigm will continue to use OOP concepts and tools as the backbone. 
# Try to get as much as you can, although you might not consider yourself a programmer. 
# These will serve you well, and makes you a more logical thinker.