#!/usr/bin/env python
# coding: utf-8

# In[1]:


from setuptools import setup


# In[2]:


setup(
    name="ginac_module",
    version="0.1",
    py_modules=[],
    package_data={
        "": ["ginac_module.cpython-311-x86_64-linux-gnu.so"],
    },
    include_package_data=True,
)


# In[ ]:




