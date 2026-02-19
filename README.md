

# 1. Import the class
from Spectrum_plot_class import RamanAnalyzer

# 2. Create an instance (give path to your data folder)
ra = RamanAnalyzer(path = "path")

# 3. Call vdep
# Replace i1, i2, i_bg, x1, x2 with your actual indices/wavelengths
bias, counts = ra.vdep(i1="LS-EL-vdep-b00001", i2="LS-EL-vdep-b00029", x1=650, x2=670, i_bg="LS-EL-vdep-b00001",save=True)
bias, counts = ra.vdep(i1="LS-EL-vdep-b00030", i2="LS-EL-vdep-b00058", x1=650, x2=670, i_bg="LS-EL-vdep-b00030", save=True)
cur, counts = ra.idep(i1="LS-EL-idep-a00001", i2="LS-EL-idep-a00020",i_bg="LS-EL-idep-a00020", x1=650, x2=670, save=True)
cur, counts = ra.idep(i1="LS-EL-idep-a00001", i2="LS-EL-idep-a00020",i_bg="LS-EL-idep-a00020", x1=610, x2=640, save=True)
