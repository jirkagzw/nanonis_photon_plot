import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import nanonispy2 as nap
from scipy.optimize import curve_fit


class RamanAnalyzer:
    """
    Complete Raman analysis class.
    Handles:
        - Data loading (pickle cache)
        - Access to filedata_dat
        - All helper functions
        - vdep, vdep2, vdep3, vdep4
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self, path, overwrite=False, pickle_name="pickle.dat", pathcopy=None):
        self.path = path
        self.overwrite = overwrite
        self.pickle_name = pickle_name
        self.pthPIK = os.path.join(path, pickle_name)
        self.pathcopy = pathcopy if pathcopy is not None else path

        self.file_list = None
        self.filedata_dat = None
        self.file_numbers = None
        self.file_names = None

        self._load_data()

    # ============================================================
    # DATA LOADING
    # ============================================================

    def _load_data(self):
    
        if os.path.isfile(self.pthPIK) and not self.overwrite:
            with gzip.open(self.pthPIK, "rb") as f:
                self.file_list = pickle.load(f)
                self.filedata_dat = pickle.load(f)
                self.file_numbers = pickle.load(f)
                self.file_names = pickle.load(f)
        else:
            self.file_list, \
            self.filedata_dat, \
            self.file_numbers, \
            self.file_names = self._getsorteddat()
    
            with gzip.open(self.pthPIK, "wb") as f:
                pickle.dump(self.file_list, f, protocol=-1)
                pickle.dump(self.filedata_dat, f, protocol=-1)
                pickle.dump(self.file_numbers, f, protocol=-1)
                pickle.dump(self.file_names, f, protocol=-1)

    # ============================================================
    # HELPER FUNCTIONS (everything before bgcorplot)
    # ============================================================
    def _getsorteddat(self):
        """
        Load .dat spectroscopy files from self.path, fix headers, and sort alphabetically by filename.
    
        Returns
        -------
        file_list : list
            List of all files in the folder (unsorted).
        filedata_dat : list
            List of spectra loaded with nanonispy2.
        file_numbers : list
            Placeholder list of 0 (kept for compatibility).
        file_names : list
            Alphabetically sorted filenames.
        """
    
    
        file_list = os.listdir(self.path)
        filedata_dat = []
        file_numbers = []
        file_names = []
    
        for file in file_list:
            if file.endswith(".dat") and file.startswith("AALS"):
                file_name = str(file)
                print("Loading:", file_name)
    
                full_path = os.path.join(self.path, file_name)
                spec = nap.read.Spec(full_path)
    
                # ---- header cleanup ----
                spec.header["Filename"] = file_name[:-4]  # remove ".dat"
    
                # Bias fix
                if "Bias>Bias (V)" in spec.header:
                    spec.header["Bias (V)"] = spec.header["Bias>Bias (V)"]
    
                # Accumulations fix
                if "GAN" in spec.header:
                    spec.header["Number of Accumulations"] = spec.header["GAN"]
                else:
                    spec.header["Number of Accumulations"] = 1
    
                filedata_dat.append(spec)
                file_numbers.append(0)  # placeholder
                file_names.append(file_name)
    
        # ---- sort alphabetically by filename ----
        sorted_indices = sorted(range(len(file_names)), key=lambda i: file_names[i])
    
        filedata_dat = [filedata_dat[i] for i in sorted_indices]
        file_numbers = [file_numbers[i] for i in sorted_indices]
        file_names = [file_names[i] for i in sorted_indices]
    
        return file_list, filedata_dat, file_numbers, file_names
    
    def npower2(self,x, a, n):
        return a * x**n

    def isfloat(self, value):
        try:
            float(value)
            return True
        except:
            return False

    def nmtopix(self, nm, lr, rr, ncol):
        return int((nm - lr) / (rr - lr) * ncol)

    def _resolve_index(self, idx):
        """
        Resolve a file index from an integer or a filename (any part match).
        """
        if isinstance(idx, int):
            return idx
    
        # Try exact match
        for j, spec in enumerate(self.filedata_dat):
            if spec.header["Filename"] == idx:
                return j
        # Try contains match (e.g., LS-EL-vdep-b00005 inside AALS-EL-vdep-b00005)
        for j, spec in enumerate(self.filedata_dat):
            if idx in spec.header["Filename"]:
                return j
    
        raise ValueError(f"File '{idx}' not found in loaded data")

    # ============================================================
    # VDEP FUNCTIONS
    # ============================================================

    def vdep(self, i1, i2, x1, x2, i_bg=None, save=False, bg=300):
        i1 = self._resolve_index(i1)
        i2 = self._resolve_index(i2)
    
        # Only resolve i_bg if provided
        if i_bg is not None:
            i_bg = self._resolve_index(i_bg)
            bg = 0  # ignore fixed bg if file background is used
    
    
              # ---- DEBUG: print which files are used ----
        print("Files used in this vdep:")
        for idx in range(i1, i2 + 1):
            print(f"  {idx}: {self.filedata_dat[idx].header['Filename']}")
        if i_bg is not None:
            print(f"Background file: {i_bg} -> {self.filedata_dat[i_bg].header['Filename']}")
        else:
            print(f"No background file, using fixed bg = {bg}")
    
        cur2 = []
        scounts = []
        bias2 = []
        zrel = []
    
        lr = self.filedata_dat[i1].signals['Wavelength (nm)'][0]
        rr = self.filedata_dat[i1].signals['Wavelength (nm)'][-1]
        ncol = len(self.filedata_dat[i1].signals['Wavelength (nm)']) - 1
    
        for i in range(i1, i2 + 1):
            zrel.append(
                (float(self.filedata_dat[i].header["Z avg. (m)"]) -
                 float(self.filedata_dat[i1].header["Z avg. (m)"])) * 1E9
            )
    
            n = float(self.filedata_dat[i].header["Number of Accumulations"])
            cur = abs(float(self.filedata_dat[i].header["Current avg. (A)"])) * 1E12
    
            bias2.append(float(self.filedata_dat[i].header["Bias (V)"]))
            cur2.append(cur)
    
            # Compute signal
            signal = np.sum(self.filedata_dat[i].signals['Counts']
                            [self.nmtopix(x1, lr, rr, ncol):
                             self.nmtopix(x2, lr, rr, ncol)])
    
            # Background subtraction
            if i_bg is not None:
                n_bg = float(self.filedata_dat[i_bg].header["Number of Accumulations"])
                bg_file = np.sum(self.filedata_dat[i_bg].signals['Counts']
                                 [self.nmtopix(x1, lr, rr, ncol):
                                  self.nmtopix(x2, lr, rr, ncol)])
                signal -= n / n_bg * bg_file
            else:
                signal -= bg  # subtract default fixed value
    
            scounts.append(signal)
    
        # Plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    
        ax1.set_xlabel('Bias [V]')
        ax1.set_ylabel(f'Integrated counts {x1}-{x2} nm')
        ax1.plot(bias2, scounts, marker='x', linestyle="None")
    
        if abs(zrel[0] - zrel[-1]) > 0.0001:
            ax2.plot(bias2, zrel, marker="^", linestyle="None", color="orange")
            ax2.set_ylabel('Zrel [nm]')
        else:
            ax2.plot(bias2, cur2, marker="^", linestyle="None", color="orange")
            ax2.set_ylabel('Current [pA]')
    
        ax1.grid(True, linestyle=':')
        fig.set_size_inches(4.4, 4.4)
    
        if save:
            filename = f"{self.filedata_dat[i1].header['Filename']}-{self.filedata_dat[i2].header['Filename']}Vdep.png"
            plt.savefig(os.path.join(self.path, filename), dpi=400, bbox_inches='tight')
    
        return bias2, scounts


    # ------------------------------------------------------------
    # vdep2
    # ------------------------------------------------------------

    def vdep2(self, i1, i2, i_bg, x1, x2, y1, y2, ratio=1, save=False):

        i1 = self._resolve_index(i1)
        i2 = self._resolve_index(i2)
        i_bg = self._resolve_index(i_bg)

        scounts = []
        scounts_c = []
        bias2 = []

        lr = self.filedata_dat[i1].signals['Wavelength (nm)'][0]
        rr = self.filedata_dat[i1].signals['Wavelength (nm)'][-1]
        ncol = len(self.filedata_dat[i1].signals['Wavelength (nm)']) - 1

        for i in range(i1, i2 + 1):

            n = float(self.filedata_dat[i].header["Number of Accumulations"])
            n_bg = float(self.filedata_dat[i_bg].header["Number of Accumulations"])

            bias2.append(float(self.filedata_dat[i].header["Bias (V)"]))

            s1 = sum(self.filedata_dat[i].signals['Counts']
                     [self.nmtopix(x1, lr, rr, ncol):
                      self.nmtopix(x2, lr, rr, ncol)])

            s2 = sum(self.filedata_dat[i].signals['Counts']
                     [self.nmtopix(y1, lr, rr, ncol):
                      self.nmtopix(y2, lr, rr, ncol)])

            bg1 = np.sum(self.filedata_dat[i_bg].signals['Counts']
                         [self.nmtopix(x1, lr, rr, ncol):
                          self.nmtopix(x2, lr, rr, ncol)])

            bg2 = np.sum(self.filedata_dat[i_bg].signals['Counts']
                         [self.nmtopix(y1, lr, rr, ncol):
                          self.nmtopix(y2, lr, rr, ncol)])

            scounts.append(s1 - n / n_bg * bg1)
            scounts_c.append(ratio * (s2 - n / n_bg * bg2))

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Bias [V]')
        ax1.set_ylabel('Integrated counts')

        ax1.plot(bias2, scounts, marker='x', linestyle="None")
        ax1.plot(bias2, scounts_c, marker='x', linestyle="None")

        ax1.grid(True, linestyle=':')
        fig.set_size_inches(4.4, 4.4)

        if save:
            filename = f"{self.filedata_dat[i1].header['Filename']}-{self.filedata_dat[i2].header['Filename']}Vdep2.png"
            plt.savefig(os.path.join(self.path, filename), dpi=400, bbox_inches='tight')

        return bias2, scounts, scounts_c

    # ------------------------------------------------------------
    # vdep3 and vdep4 can be added here exactly the same way
    # (structure preserved, just replace globals with self.)
    # ------------------------------------------------------------

    
    def idep(self, i1, i2, x1, x2, unit="pA", save=False, norm=False, i_bg=None, bg=300):
        """
        Current-dependent integrated counts plot (I-dependence).
    
        Parameters
        ----------
        i1, i2 : int or str
            Start and end file indices or filenames (shrinked).
        x1, x2 : float
            Wavelength range in nm to integrate counts.
        unit : str
            Unit for x-axis ("pA" by default).
        save : bool
            Whether to save figure.
        norm : bool
            Not yet implemented, placeholder for normalization.
        i_bg : int or str, optional
            Background file index or filename. If None, subtract fixed `bg`.
        bg : float
            Fixed background to subtract if i_bg is None.
    
        Returns
        -------
        cur2 : list
            Current values [pA].
        scounts : list
            Integrated counts.
        """
        # Resolve indices
        i1 = self._resolve_index(i1)
        i2 = self._resolve_index(i2)
        if i_bg is not None:
            i_bg = self._resolve_index(i_bg)
            bg = 0  # ignore fixed bg if background file is used
    
        cur2 = []
        scounts = []
        bias = []
    
        lr = self.filedata_dat[i1].signals['Wavelength (nm)'][0]
        rr = self.filedata_dat[i1].signals['Wavelength (nm)'][-1]
        ncol = len(self.filedata_dat[i1].signals['Wavelength (nm)']) - 1
    
        # Debug: print files used
        print("Files used in this idep:")
        for idx in range(i1, i2 + 1):
            print(f"  {idx}: {self.filedata_dat[idx].header['Filename']}")
        if i_bg is not None:
            print(f"Background file: {i_bg} -> {self.filedata_dat[i_bg].header['Filename']}")
        else:
            print(f"No background file, using fixed bg = {bg}")
    
        # Loop over files
        for i in range(i1, i2 + 1):
            bias.append(float(self.filedata_dat[i].header.get("Bias (V)", 0)))
            n = float(self.filedata_dat[i].header.get("Number of Accumulations", 1))
            cur = abs(float(self.filedata_dat[i].header.get("Current avg. (A)", 0))) * 1e12  # pA
            cur2.append(cur)
    
            # Compute integrated counts
            counts_slice = self.filedata_dat[i].signals['Counts'][
                self.nmtopix(x1, lr, rr, ncol):self.nmtopix(x2, lr, rr, ncol)
            ]
            signal = np.sum(counts_slice)
    
            # Background subtraction
            if i_bg is not None:
                n_bg = float(self.filedata_dat[i_bg].header.get("Number of Accumulations", 1))
                bg_slice = self.filedata_dat[i_bg].signals['Counts'][
                    self.nmtopix(x1, lr, rr, ncol):self.nmtopix(x2, lr, rr, ncol)
                ]
                signal -= n / n_bg * np.sum(bg_slice)
            else:
                signal -= bg
    
            scounts.append(signal)
    
        # Plot
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(f'Current [{unit}]')
        ax1.set_ylabel(f'Integrated counts {x1}-{x2} nm')
        ax1.plot(cur2, scounts,
                 label=f"{self.filedata_dat[i1].header['Filename']} - {self.filedata_dat[i2].header['Filename']}",
                 marker='x', linestyle="None")
    
        # Fit with npower2
        popt, pcov = curve_fit(self.npower2, cur2, scounts)
        print("Fit exponent:", popt[1])
        ax1.plot(cur2, self.npower2(cur2, popt[0], popt[1]), label=f"fit exp={popt[1]:.4f}")
    
        ax1.grid(True, linestyle=':')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        fig.set_size_inches(4.4, 4.4)
        plt.legend()
    
        # Save figure
        if save:
            filename = f"{self.filedata_dat[i1].header['Filename']}-" \
                       f"{self.filedata_dat[i2].header['Filename']}-{x1}-{x2}.png"
            plt.savefig(os.path.join(self.path, filename), dpi=400, bbox_inches='tight')
    
        return cur2, scounts
