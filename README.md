# STATY
**STATY - learn, do and create data stories**

STATY provides a user-friendly interface for performing various classical statistical and machine learning methods on a user-specific data.   
STATY is an ongoing educational project designed with the aim of improving data literacy among undergraduate and graduate students.

**STATY offers two options to get started:**

* **Source Code:** This guide provides detailed instructions for installing and running STATY from the source code. 
* **Portable Version:** Information on obtaining a user-friendly, ready-to-run portable version can be found [here](https://github.com/lilulamili/STATY/wiki/STATY).


## Getting started with source code 

**1. Python Installation**    
> [!TIP]
> Make sure to check the checkbox labeled "Add Python ... to PATH" during the installation process.  
  This ensures you can easily run Python commands from your terminal later.  
   
   Install Python 3.11.8 from: https://www.python.org/downloads/release/python-3118/  
   Windows: Scroll down to the section 'Files' and select `Windows installer (64-bit)`  
   macOS: Scroll down to the section 'Files' and select `macOS 64-bit universal2 installer`

**2. VSC Installation**   
       Install Visual Studio Code by pressing the big blue button from: https://code.visualstudio.com/

**3. Download 'STATY'**   
  To download STATY, press the green button `<> Code` above and select `Download ZIP`. 
       
**4. Get 'STATY' ready**  
   Open VSCode. Go to the `File` menu, select `Open Folder`, and then navigate to the project folder you just created (the one where the file are).  
   `„Do you trust the Author of this Folder” – click “yes”`

**5. create a virtual environment**   
   Locate the `Terminal` panel in VScode (usually at the bottom of the window). If it's not visible, go to the View menu and select Terminal. Type the following command in the terminal to create a virtual environment:

   Windows: `py -3 -m venv .venv`

   macOS:`python3 -m venv .venv `  
   
   `We noticed a new virtual environment....   click "yes"`
   
   > [!TIP]
   > Don't foget the dot before the second venv `.venv`  
> Windows: In case of a policy challenge type the following command in the terminal:   
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**6. Activate the virtual environment**   
  Type the following command in the terminal:

  Windows:`.venv\Scripts\activate`

  macOS:`source .venv/bin/activate` 

**7. Install all application components**   
   Type the following command in the terminal: `pip install -r requirements.txt`

**8. Run STATY**   
   Type the following command in the terminal: `streamlit run staty.py`  
   The app will open in your default browser `http://localhost:8075/`
