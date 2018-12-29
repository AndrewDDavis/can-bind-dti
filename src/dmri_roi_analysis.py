#!/usr/bin/env python
# coding: utf8
"""This is dmri_roi_analysis.py, functions for ROI analysis of diffusion MRI data.

   Recommended import line:
       import src.dmri_roi_analysis as dmroi

   Made for the CAN-BIND project

   v0.3.1 (Sept 2018) by Andrew Davis (addavis@gmail.com)
"""

from __future__ import print_function, division
import sys, os, shutil, re
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler           # for manipulating matplotlib color cycles
from scipy import stats
from sklearn.linear_model import LinearRegression
from nipype.algorithms import icc
import statsmodels.api as sm
import multiprocessing as mp

# import dev branch of pymer4 with REML option to fit()
sys.path.insert(0, os.path.expanduser('~/Documents/Research_Projects/CAN-BIND_DTI/src/pymer4_dev'))
import src.pymer4_dev.pymer4.models as pm4dev   # explicit path to be sure which one you're getting

# import my dev of statsmodels that returns p-value of tukeyhsd
sys.path.insert(0, os.path.expanduser('~/Documents/Research_Projects/CAN-BIND_DTI/src/statsmodels_dev'))
from src.statsmodels_dev.statsmodels.stats import multicomp as mcdev    # edited to include sandbox dev version


# figure properties
plt.rcParams['font.size'] = 10.
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['axes.titlesize'] = 'medium'
plt.interactive(False)


def print_log(text, logfile):
    """Print text to stdout and also to a log file. Used in model_lmer etc."""

    print(text)

    with open(logfile, 'a') as of:
        of.write(text + "\n")


def short_name(orig_name):
    """Shorten ROI names to 40 chars for plot titles"""

    if len(orig_name) < 41:
        new_name = orig_name
    else:
        new_name = orig_name[0:32] + "..." + orig_name[-5:]

    return new_name


def plt_jitter(v, n, m=3., jitrng=None, yvals=None):
    """Generate an array for plotting; add randomness to x-plotting
    locations (v, as 1-D array) so markers don't all overlie one another. By
    default, jitters by 17% of the mean separation both ways. If given the yvals,
    tapers the x value jitter based on the SD.

    Returns 2D array with n jittered copies of v.
    """
    v = v.astype(float)

    if yvals is not None:
        # generate factor to modify jitrng based on distance from mean
        # jitrng goes to zero, linearly, at 2SD
        yscore = np.abs(yvals - np.mean(yvals))/(2*np.std(yvals))
        yscore[yscore > 1.] = 1.
        yscore = 1 - yscore

    if jitrng is None:
        jitrng = np.diff(v).mean()/m

    # make n copies of v with jitter in range jitrng
    np.random.seed(42)      # for consistency across plots
    x = np.tile(v, [n, 1])
    for i in range(n):
        if yvals is None:
            f = jitrng
        if yvals is not None:
            f = jitrng*yscore[i]

        # np.r.random gives uniform dist. values on [0,1)
        r = f*np.random.random(len(v)) - f/2.
        x[i, :] += r

    return x


def detect_outliers(y, zs_thr=None, sided='both', report_thresh=True):
    """Accepts an array, *y*, and detects outliers in the data based on the
    MAD. zs_thr is the z-score threshold, by default calculated based on
    the expectation value of 0.5 data points. sided can be ['both', 'high', 'low'],
    determining whether outliers are considered on each side of the distribution.

    Returns:
      - outl_bool; bool array with the same dimesions as *y*, with False values
        corresponding to outliers.
      - zs_thr; calculated z-score threshold as a scalar
      - sd_rob; robust estimation of stdev via the MAD
    """

    assert y.ndim == 1

    if zs_thr is None:
        # find the z-score where the expectation value is 0.5 data points,
        #   in either tail. Use this as threshold, since beyond that we
        #   expect to find 0 data points for this data set.
        #   old: zs_thr = 3.43 <- this was 0.49 data points on 808 values
        n = len(y)
        ntails = 2
        p0 = 0.5/n
        zs_thr = stats.norm.ppf(1.0 - p0/ntails)
        if report_thresh:
            print("Calculated z-score thresh for E(0.5): {:0.2f}".format(zs_thr))

    # calculate robust stdev from median absolute deviation
    ymed = np.median(y)
    ymad = np.median(np.abs(y - ymed))
    sd_rob = ymad*1.4826

    if sided == 'both':
        # consider outliers to both sides of the distribution
        outl_bool = np.abs(y - y.mean()) > zs_thr*sd_rob
    elif sided == 'high':
        outl_bool = y > y.mean() + zs_thr*sd_rob
    elif sided == 'low':
        outl_bool = y < y.mean() - zs_thr*sd_rob

    return outl_bool, zs_thr, sd_rob


def plot_hist_kde(y, xlabel, title='', groups=('all',),
                  bools=None, clrs=('g',)):
    """Plot a histogram and KDE of input values, with option to
    split up into groups for outlier determination.

    Returns a figure object."""

    if bools is None:
        bools = ([True]*len(y),)
    elif isinstance(bools, dict):
        dict_of_bools = bools.copy()
        bools = []
        for g in groups:
            bools.append(dict_of_bools[g])


    # set up figure
    fig, ax1 = plt.subplots(figsize=(5.5, 4), dpi=150)
    # ax1 = target_ax
    # fig = ax1.get_figure()

    # plot histogram and make axes for density
    _, _, patches = ax1.hist(y, bins=20)

    ax2 = ax1.twinx()
    xlims = ax1.get_xlim()
    x = np.linspace(xlims[0], xlims[1], 500)

    # plot outlier thresholds and gaussian kdes per group
    outl_ids = []
    outl_patch_cnt = np.zeros(len(patches))

    for g, tf, clr in zip(groups, bools, clrs):

        # plot vertical lines at outlier thresholds
        outl_bool, zs_thr, sd_rob = detect_outliers(y[tf], report_thresh=False)

        vline_kwargs = {'color': clr, 'ls': ":", 'alpha': 0.67}
        uthresh = y[tf].mean() + zs_thr*sd_rob
        lthresh = y[tf].mean() - zs_thr*sd_rob
        ax1.axvline(uthresh, **vline_kwargs)
        ax1.axvline(lthresh, **vline_kwargs)

        # Report outlier subjids
        if np.any(outl_bool) and isinstance(outl_bool, pd.Series):
            new_outl_ids = y[tf].index[outl_bool].tolist()
            print("Outlier subjids found beyond z={:4.2f} (grp={}): {}".format(zs_thr, g, new_outl_ids))
            outl_ids.extend(new_outl_ids)

            # Change colour of outlier patches
            #   a bit tricky because patches may straddle the outlier line
            #   and outliers could be on either side

            # find the bin of the outl_subjid and change the colour of that patch
            w0 = patches[0].get_x()
            w = patches[0].get_width()
            for oid in new_outl_ids:
                oval = y[oid]
                obin = int(np.floor((oval - w0)/w))

                if obin >= len(patches):
                    obin = len(patches) - 1
                elif obin < 0:
                    obin = 0

                opatch = patches[obin]
                outl_patch_cnt[obin] += 1

                # be careful not to avoid outlier patches already drawn
                newy = opatch.get_height() - outl_patch_cnt[obin]
                newpatch = plt.Rectangle((opatch.get_x(), newy), w, 1., ec='None')

                # make a maroon patch if outside lines, purple if on the line
                if (opatch.get_x() > uthresh) or (opatch.get_x() + opatch.get_width() < lthresh):
                    newpatch.set_facecolor('maroon')
                elif (opatch.get_x() + opatch.get_width() > uthresh) or (opatch.get_x() < lthresh):
                    newpatch.set_facecolor('purple')

                ax1.add_patch(newpatch)


        # plot smooth kde to view limits
        kde = stats.gaussian_kde(y[tf])
        ax2.plot(x, kde(x), color=clr, ls='-', alpha=0.5)
        ax2.set_xlim(xlims)

    # labels, make histogram on top, show, etc
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Density")
    ax2.set_ylim(bottom=0.)
    ax1.set_title(title)
    # ax1.set_zorder(ax2.get_zorder()+1)
    # ax1.patch.set_visible(False)

    fig.tight_layout()
    fig.canvas.draw()

    return fig, outl_ids


def run_modl_fit(tup):
    """Run model fit method (defined at top level for parallelization with mp)"""
    # modl.fit(summarize=False)

    modl, logfile = tup

    lfh = open(logfile, 'a')
    sys.stdout = lfh        # write fit error messages to log for later parsing

    if 'twogroup' in modl.formula:
        modl.fit(factors={'twogroup': ['ctrl', 'trmt']}, summarize=False)
    elif 'trigroup' in modl.formula:
        modl.fit(factors={'trigroup': ['ctrl', 'resp', 'nonr']}, summarize=False)

    sys.stdout = sys.__stdout__
    lfh.close()
    return modl


def LLR_test(tup):
    """Likelihood ratio test (uses Log(Likelihood) values) from 2 models with
    different fixed effects. Significant p-value indicates the fixed effect is
    significant.
    """

    # See Galecki2013; Linear Mixed Effects Models Using R
    #   and Gumedze2011.
    # Note should use ML method to fit the data in this case (REML=False)

    null_model_formula, alt_model_formula, data, dfdiff, logfile = tup

    lfh = open(logfile, 'a')
    sys.stdout = lfh        # write fit error messages to log for later parsing

    null_model = pm4dev.Lmer(null_model_formula, data=data)
    null_model.fit(REML=False, summarize=False)

    alt_model = pm4dev.Lmer(alt_model_formula, data=data)
    alt_model.fit(REML=False, summarize=False)

    LL_null = null_model.logLike
    LL_alt = alt_model.logLike

    D = 2*(LL_alt - LL_null)
    p = stats.chi2.sf(D, dfdiff)  # where dfdiff is difference in model DF, usually 1

    sys.stdout = sys.__stdout__
    lfh.close()
    return p


def cohen_d(x1, x2):
    """Calculate Cohen's d, a standardized measure of effect size for
    group differences. A ref for this method of pooling standard deviations
    is Lakens2013, DOI=10.3389/fpsyg.2013.00863.
    """

    # pooled standard deviation
    n1, n2 = len(x1), len(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s = np.sqrt(((n1-1)*v1 + (n2-1)*v2)/float(n1 + n2 - 2))

    u1, u2 = np.mean(x1), np.mean(x2)

    # standardized difference measure, cohen's d
    return (u1 - u2)/s


def lmer_results_summary(roidata_obj, write_combo=True):
    """Create a table of all LMER FDR-p results available so far in the output
    directory.

    Single required argument can be an RoiData object, or just a directory.

    These are FDR corrected results coming from the LLR test values in the
    LME model.

    Summary files are written out as lmer_fdr_combined_results_df.pkl and
    lmer_fdr_combined_results.csv.
    """

    if roidata_obj.__class__.__name__ == 'RoiData':
        arg_is_obj = True
        roidata_dir = roidata_obj.roidata_dir
    else:
        arg_is_obj = False
        roidata_dir = roidata_obj

    # N.B.
    # logfile = roidata_obj.roidata_dir + "lmer_fdr_" + roidata_obj.metric + "_log.txt"
    # lmer_plot_dir = roidata_obj.roidata_dir + "LMER_plots/"
    # e.g. roidata_dir: '~/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03/'

    lmer_model_logfiles = glob(roidata_dir + 'lmer_model_??_log.txt')
    lmer_fdr_logfiles = glob(roidata_dir + 'lmer_fdr_??_log.txt')
    restable_files = glob(roidata_dir + "lmer_fdr_??_results_df.pkl")

    assert (len(lmer_fdr_logfiles) > 0), "No FDR logfiles found in {}.".format(roidata_dir)

    # Search the logfiles for "Any effects" = True lines and model convergence failures
    print("Searching logfiles for any significant effects...")
    sig_effects_df = pd.DataFrame(data=np.nan,
                                  index=['FA', 'MD', 'L1', 'RD'],
                                  columns=['LR_group', 'LR_time', 'LR_gxt', '3G_rxt', 'fit_error'])

    for lmer_fdr_logfile in lmer_fdr_logfiles:
        metric = lmer_fdr_logfile[-10:-8]

        with open(lmer_fdr_logfile, 'r') as lf:
            for line in lf:
                if re.search('Any.*True', line):
                    # print(line, end='')
                    effect = line.split(' ')[3]
                    sig_effects_df.loc[metric, effect] = True

    # Also search for any fit errors
    for logfile in lmer_model_logfiles + lmer_fdr_logfiles:
        fn = logfile.split('/')[-1]
        metric = fn.split('_')[2]

        with open(logfile, 'r') as lf:
            for line in lf:
                if re.search('failed to converge', line):
                    # check for error message:
                    #   unable to evaluate scaled gradient
                    #   Model failed to converge: degenerate  Hessian with 1 negative eigenvalues
                    sig_effects_df.loc[metric, 'fit_error'] = True

    print(sig_effects_df)
    print("")

    if arg_is_obj:
        print("Excluded ROIs:\n"
            + "\n".join(roidata_obj.dropped_rois) + "\n")


    # Generate main table from all metric results tables
    if write_combo:
        results_tables = []
        for rtf in restable_files:
            results_tables.append(pd.read_pickle(rtf))

        combined_results = pd.concat(results_tables, axis=0, ignore_index=True)
        combined_results.sort_values(by=['ROI', 'Groups', 'Metric'], axis='index', inplace=True)
        # combined_results.set_index(['ROI', 'Groups', 'Metric'], inplace=True, verify_integrity=True)

        # Write out the combined results
        combined_results.to_pickle(roidata_dir + "lmer_fdr_combined_results_df.pkl")
        combined_results.to_csv(roidata_dir + "lmer_fdr_combined_results.csv", index=False, encoding='utf-8')


        # Clean up combined results to report only LLR_FDR-p < 0.05 and HSD == True
        reportable_results = combined_results.drop(index=combined_results.index[combined_results['LLR_FDR-p']>0.05])
        reportable_results.drop(index=reportable_results.index[~reportable_results['Also_HSD']], inplace=True)
        reportable_results['ROI'] = reportable_results['ROI'].apply(abbrev_roi)

        # Create Sig col, indicating high significance with stars
        sig_col = np.array(['']*reportable_results.shape[0], dtype='|S3')
        sig_col[reportable_results['Grp_FDR-p'] < 0.01] = '**'
        sig_col[reportable_results['Grp_FDR-p'] < 0.001] = '***'
        reportable_results['Sig'] = sig_col

        # Two sig figs for cohen_d but 3 for Delta
        reportable_results['Delta'] = reportable_results['Delta'].map(lambda x: '{: 6.3f}'.format(x))
        reportable_results['Cohen_d'] = reportable_results['Cohen_d'].map(lambda x: '{: 5.2f}'.format(x))

        # Report L1 as AD
        tf = (reportable_results['Metric'] == 'L1')
        reportable_results.loc[tf, 'Metric'] = 'AD'

        # Clean up cols no longer needed
        reportable_results.drop(columns=['LLR_FDR-p', 'Also_HSD', 'Grp_FDR-p'], inplace=True)

        # Write out; not using float_format='% 5.3f', since it's taken care of above
        reportable_results.to_csv(roidata_dir + 'lmer_fdr_reportable_results-plt05_HSD.csv', index=False, encoding='utf-8')


def abbrev_roi(roi, fwd_lookup=True):
    """Accepts ROI name, returns abbreviation. Set roi=None to get list
    of all ROIs.

    Set fwd_lookup=False to get full name from abbreviation.
    """

    roi_abbrevs = {'Genu_of_corpus_callosum': 'GCC',
                   'Body_of_corpus_callosum': 'BCC',
                   'Splenium_of_corpus_callosum': 'SCC',
                   'Fornix_column_and_body_of_fornix': 'FX',
                   'Superior_cerebellar_peduncle_R': 'SCP-R',
                   'Superior_cerebellar_peduncle_L': 'SCP-L',
                   'Cerebral_peduncle_R': 'CP-R',
                   'Cerebral_peduncle_L': 'CP-L',
                   'Anterior_limb_of_internal_capsule_R': 'ALIC-R',
                   'Anterior_limb_of_internal_capsule_L': 'ALIC-L',
                   'Posterior_limb_of_internal_capsule_R': 'PLIC-R',
                   'Posterior_limb_of_internal_capsule_L': 'PLIC-L',
                   'Retrolenticular_part_of_internal_capsule_R': 'RLIC-R',
                   'Retrolenticular_part_of_internal_capsule_L': 'RLIC-L',
                   'Anterior_corona_radiata_R': 'ACR-R',
                   'Anterior_corona_radiata_L': 'ACR-L',
                   'Superior_corona_radiata_R': 'SCR-R',
                   'Superior_corona_radiata_L': 'SCR-L',
                   'Posterior_corona_radiata_R': 'PCR-R',
                   'Posterior_corona_radiata_L': 'PCR-L',
                   'Posterior_thalamic_radiation_include_optic_radiation_R': 'PTR-R',
                   'Posterior_thalamic_radiation_include_optic_radiation_L': 'PTR-L',
                   'Sagittal_stratum_include_inferior_longitidinal_fasciculus_and_inferior_fronto-occipital_fasciculus_R': 'SS-R',
                   'Sagittal_stratum_include_inferior_longitidinal_fasciculus_and_inferior_fronto-occipital_fasciculus_L': 'SS-L',
                   'External_capsule_R': 'EC-R',
                   'External_capsule_L': 'EC-L',
                   'Cingulum_cingulate_gyrus_R': 'CgC-R',
                   'Cingulum_cingulate_gyrus_L': 'CgC-L',
                   'Cingulum_hippocampus_R': 'CgH-R',
                   'Cingulum_hippocampus_L': 'CgH-L',
                   'Fornix_cres_Stria_terminalis_can_not_be_resolved_with_current_resolution_R': 'FX/ST-R',
                   'Fornix_cres_Stria_terminalis_can_not_be_resolved_with_current_resolution_L': 'FX/ST-L',
                   'Superior_longitudinal_fasciculus_R': 'SLF-R',
                   'Superior_longitudinal_fasciculus_L': 'SLF-L',
                   'Superior_fronto-occipital_fasciculus_could_be_a_part_of_anterior_internal_capsule_R': 'SFO-R',
                   'Superior_fronto-occipital_fasciculus_could_be_a_part_of_anterior_internal_capsule_L': 'SFO-L',
                   'Uncinate_fasciculus_R': 'UNC-R',
                   'Uncinate_fasciculus_L': 'UNC-L',
                   'Tapetum_R': 'TAP-R',
                   'Tapetum_L': 'TAP-L',
                   'Skel_periph_R': 'SP-R',
                   'Skel_periph_L': 'SP-L',
                   'Skel_mean': 'Skel-mean'}

    if roi is None:
        from pprint import pprint
        pprint(roi_abbrevs)
    elif fwd_lookup:
        return roi_abbrevs[roi]
    else:
        reversed_abbrevs = {v:k for k,v in roi_abbrevs.items()}
        return reversed_abbrevs[roi]


class RoiData:
    """Holds the attributes and methods associated with ROI analysis of
    DMRI data for the CAN-BIND-01 data set on Erikson.
    """

    # Class variables shared by all instances
    proj_dir = os.path.expanduser("~/Documents/Research_Projects/CAN-BIND_DTI/")


    # Cohort csvs contain info on subject group, age, etc
    chrt01_df = pd.read_csv(proj_dir + "group_results/cohort_csvs/cohort-v01.csv",
                            header=0,
                            index_col=0)

    # Standard list of subprojects in the correct order, including QNS_G and UBC_G
    subprojs = np.loadtxt(proj_dir + "group_results/cohort_csvs/subprojs-list.txt", dtype='|S5')


    # Instantiate the object
    def __init__(self, visit_code, metric):
        """Each instance of the object is defined by what data will be
        included (FA, MD, etc), and what visits will be included
        (v01, v01v02v03, etc).
        """
        # instance variable unique to each instance
        self.visit_code = visit_code
        self.metric = metric
        if metric == "L1":
            self.metric_label = r"AD ($\mu$m$^2$/ms)"    # 10^-3 mm^2/s = 1 um^2/ms
        elif metric in ["MD", "RD"]:
            self.metric_label = metric + r" ($\mu$m$^2$/ms)"
        else:
            self.metric_label = metric

        # Parse visit code
        gr_res_dir = RoiData.proj_dir + "group_results/"

        if visit_code == "1":
            self.roidata_dir = gr_res_dir + "merged_skel_v01/"

            chrt_csv = gr_res_dir + "cohort_csvs/cohort-v01.csv"
            visit_list = gr_res_dir + "cohort_csvs/visit_list-v01.txt"

        elif visit_code == "123":
            self.roidata_dir = gr_res_dir + "merged_skel_v01v02v03/"

            chrt_csv = gr_res_dir + "cohort_csvs/cohort-v01v02v03.csv"
            visit_list = gr_res_dir + "cohort_csvs/visit_list-v01v02v03.txt"
        else:
            raise ValueError("Unknown visit_code")

        # Visit code specific data -- cohort info and ROI data
        # self.visit_ser = pd.read_csv(visit_list, header=None, index_col=0, squeeze=True)
        self.chrt_df = pd.read_csv(chrt_csv, header=0, index_col=0)    # 308 x 8 for "all" data set

        JHU_data_df = pd.read_csv(self.roidata_dir + "JHU-40ROI_skel_" + self.metric + "_data.csv",
                                  header=0,
                                  index_col=0)  # 808 x 40 for "all" data set
        skel_periphR_ser = pd.read_csv(self.roidata_dir + "skelperiph_R_mean" + self.metric + "_data.csv",
                                       header=None,
                                       names=["SUBJLABEL", "Skel_periph_R"],
                                       index_col=0,
                                       squeeze=True)
        skel_periphL_ser = pd.read_csv(self.roidata_dir + "skelperiph_L_mean" + self.metric + "_data.csv",
                                       header=None,
                                       names=["SUBJLABEL", "Skel_periph_L"],
                                       index_col=0,
                                       squeeze=True)
        skel_mean_ser = pd.read_csv(self.roidata_dir + "skel_mean" + self.metric + "_data.csv",
                                    header=None,
                                    names=["SUBJLABEL", "Skel_mean"],
                                    index_col=0,
                                    squeeze=True)

        # Combine ROI data into one 43-column dataframe
        assert(np.all(JHU_data_df.index == skel_mean_ser.index)
           and np.all(JHU_data_df.index == skel_periphR_ser.index)
           and np.all(JHU_data_df.index == skel_periphL_ser.index))
        self.roi_df = pd.concat([JHU_data_df, skel_periphR_ser, skel_periphL_ser, skel_mean_ser], axis=1)


        # QC Info from pre-processing -- generate a reject df that matches roi_df
        qc_summary_df = pd.read_csv(RoiData.proj_dir + "group_results/QC/Master_QC_Summary.csv",
                                    header=1,
                                    index_col=False)
        qc_summary_df.set_index(qc_summary_df['subjid'] + "_0" + qc_summary_df['visit'].astype(str), inplace=True)
        qc_summary_df.drop(columns=['subjid', 'visit'], inplace=True)

        self.qc_reject_ser = qc_summary_df.loc[self.roi_df.index, 'reject']
        self.qc_rejects_excluded = False


        # Metric specific paths
        self.skelmean01_csv = gr_res_dir + "merged_skel_v01/skel_mean" + self.metric + "_data.csv"
        # self.model_regr_npy = gr_res_dir + "merged_skel_v01/regrs_sex_age_" + self.metric + ".npy"


        # Set up some attributes to test for completed steps later
        self.subj_bools_data = None
        self.regrs_sa = None
        self.roi_saresid_df = None
        self.regrs_site_effects = None
        self.roi_gs_df = None
        self.dropped_rois = []
        self.sse_outl_df = None
        self.dist_outl_dfs = None
        self.imputed_roi_df = None
        self.imputed_roi_gs_df = None
        self.lmer_pvals = None


    def run_pipeline(self, pipln):
        """Run a pre-determined set of steps to analyze the data.
        Possible pipelines:
          - incl_outl : does not exclude any outliers (neither SSE nor dist routines are run)
          - excl_sse_outl : excludes SSE outliers but does not run dist outl routine
          - excl_outl : runs both outl steps
          - excl_outl_tothemax : further excludes *subjects* that have mean SSE outlier, runs dist outliers twice
        """

        assert pipln in ['incl_outl', 'excl_sse_outl', 'excl_outl', 'excl_outl_tothemax']
        self.pipeline_run = pipln   # record for later object introspection

        # Define colours to print headings and clear terminal screen
        header_style = '\x1b[0;30;46m'
        bold_style = '\x1b[1;30;46m'
        cancel_style = '\x1b[0m'
        clear_screen = '\x1b[2J'
        def header_print(s):
            """Print pipeline header messages that stand out in the terminal text."""
            # print(clear_screen)  # too much wasted space
            print("\n\n")
            print(header_style + "Running " + bold_style + s + header_style + " ..." + cancel_style)


        # Generate boolean vars that match the cohorts and data for groups and subprojs
        header_print("gen_databools")
        self.gen_databools()

        # Permanently exclude QC failures from dfs and bools
        header_print("remove_qc_rejects")
        self.remove_qc_rejects()

        # Drop skel_mean since it's redundant and FX because it's got partial volume from CSF
        #   Note this just identifies the ROIs it's up to the methods to actually drop them
        #   e.g. regr_model_site_effects is one of the first affected
        header_print("drop_rois")
        self.drop_rois(['Skel_mean', 'Fornix_column_and_body_of_fornix'])

        if (self.visit_code == '123') and (pipln in ['excl_sse_outl', 'excl_outl', 'excl_outl_tothemax']):
            # Identify outlier ROIs per subject based on poor fits (SSE-based outliers)
            # biological values: change of 0.05 in FA over 6 years in development (krogsrud 2016)
            # what is my mean temporal change?  See fixed effects, mean temporal change of responders, calculate MS
            if pipln == 'excl_outl_tothemax':
                header_print("calc_subj_sse with excl_outl_subjids=True")
                self.calc_subj_sse(view=False, interactive=False, excl_outl_subjids=True)
            else:
                header_print("calc_subj_sse")
                self.calc_subj_sse(view=False, interactive=False)

        # Process data for site effects
        header_print("regr_model_sa")
        if (pipln == 'incl_outl') or (self.visit_code != '123'):
            self.regr_model_sa(ignore_SSE_outliers=False, view=False)
        else:
            self.regr_model_sa(view=False)          # model sex & age within a subproject using CTRL v01 skel-mean data

        header_print("remove_sa_effects")
        self.remove_sa_effects()                    # takes self.roi_df, creates self.roi_saresid_df; ignores sse-outliers

        header_print("regr_model_site_effects")
        if (pipln == 'incl_outl') or (self.visit_code != '123'):
            self.regr_model_site_effects(ignore_SSE_outliers=False, view=False)
        else:
            self.regr_model_site_effects(view=False)    # takes self.roi_saresid_df, calculates subproj effect regression coefs across ROIs from CTRL v01 values; ignores SSE-outliers. 123 R^2 range: 0.93 -> 0.99

        header_print("remove_site_effects")
        self.remove_site_effects()                  # takes self.roi_df, creates self.roi_gs_df

        if (self.visit_code == '123') and (pipln in ['excl_sse_outl', 'excl_outl', 'excl_outl_tothemax']):
            header_print("impute_outliers with outl_type='SSE'")
            self.impute_outliers(outl_type='SSE')

        if pipln in ['excl_outl', 'excl_outl_tothemax']:
            # Identify dist-based outliers
            header_print("calc_gs_dists")
            self.calc_gs_dists(view=False)
            header_print("impute_outliers with outl_type='dist'")
            self.impute_outliers(outl_type='dist')

            if pipln == 'excl_outl_tothemax':
                header_print("calc_gs_dists (again)")
                self.calc_gs_dists(view=False)
                header_print("impute_outliers with outl_type='dist'")
                self.impute_outliers(outl_type='dist')

        # Big plots for figure in paper
        header_print("plot_raw_gs")
        self.plot_raw_gs(sse_outl='ignore', dist_outl='ignore', use_imputed='GS', view=False)

        # ICC
        if self.visit_code == '123':
            header_print("calc_icc")
            self.calc_icc(view=False)

        # Stats
        if self.visit_code == '1':
            header_print("test_grpdiff_bsln")
            self.test_grpdiff_bsln()

        elif self.visit_code == '123':
            # for curiosity
            header_print("test_grpdiff_bsln")
            self.test_grpdiff_bsln()
            header_print("test_respdiff_wk8")
            self.test_respdiff_wk8()

            header_print("model_lmer")
            self.model_lmer()
            header_print("lmer_fdr")
            self.lmer_fdr(view=False)


    def gen_databools(self):
        """Find boolean arrays matching the data (i.e. the index of roi_df) to
        pick out the controls and each subproject."""
        index_data = self.roi_df.index
        N_data = len(index_data)

        # Controls
        ctrl_bool__chrt = self.chrt_df['Group'] == "Control"
        ctrl_subjids = self.chrt_df.index[ctrl_bool__chrt]

        ctrl_bool__data = pd.Series(data=np.full(N_data, False), index=index_data)    # len=702 for "123" cohort
        for sid in ctrl_subjids:
            sid_bool = index_data.str.startswith(sid)
            ctrl_bool__data = ctrl_bool__data | sid_bool

        # Responders
        resp_bool__chrt = self.chrt_df['Respond_WK8'] == "Responder"
        resp_subjids = self.chrt_df.index[resp_bool__chrt]

        resp_bool__data = pd.Series(data=np.full(N_data, False), index=index_data)
        for sid in resp_subjids:
            sid_bool = index_data.str.startswith(sid)
            resp_bool__data = resp_bool__data | sid_bool

        # Non-responders
        nonr_bool__chrt = self.chrt_df['Respond_WK8'] == "NonResponder"
        nonr_subjids = self.chrt_df.index[nonr_bool__chrt]

        nonr_bool__data = pd.Series(data=np.full(N_data, False), index=index_data)
        for sid in nonr_subjids:
            sid_bool = index_data.str.startswith(sid)
            nonr_bool__data = nonr_bool__data | sid_bool

        # Some subjects with v01 data do *not* have resp/nonr information at wk8.
        # These will be excluded from later group tests involving resp/nonr status.
        # So the TRMT group is larger than RESP+NONR. The only negative consequence
        #   that I can see is that subjs in trmt but not resp/nonr don't get
        #   considered for outlier exclusion in calc_gs_dists.


        # Subprojects
        def find_sp_ids(sp_label):
            """Find bool for subjids in subproj matching data df"""
            # pick the subproj subjids from cohort df
            subproj_bool__chrt = self.chrt_df.Subproj == sp_label
            subjids_in_subproj = self.chrt_df.index[subproj_bool__chrt]

            # pick the data rows corresponding to subjids in subproj
            subproj_bool__data = pd.Series(data=np.full(N_data, False), index=index_data)
            for sid in subjids_in_subproj:
                sid_bool = index_data.str.startswith(sid)
                subproj_bool__data = subproj_bool__data | sid_bool

            return subproj_bool__data

        subproj_bools__data = []    # will have 11 values after appends below
        for sp_label in RoiData.subprojs:
            print("Looking at {}".format(sp_label))

            subproj_bools__data.append(find_sp_ids(sp_label))

        print("")


        # Assign attributes to class instance in dicts or list
        print("Assigning bool atributes: subj_bools_chrt, subj_bools_data, subproj_bools_data\n")
        self.subj_bools_chrt = {'ctrl': ctrl_bool__chrt,
                                'resp': resp_bool__chrt,
                                'nonr': nonr_bool__chrt}
        self.subj_bools_data = {'ctrl': ctrl_bool__data,
                                'resp': resp_bool__data,
                                'nonr': nonr_bool__data}
        self.subproj_bools__data = subproj_bools__data


    def remove_qc_rejects(self):
        """Remove QC rejected values from the data set. This may be called from
        the plotting method or the stat test methods.

        QC rejected visits apply to the whole visit data, all rois. In 123
          visit_code data sets, this removes the other two visits as well.

        Rejected visits should be removed from data dfs and bools:
          self.roi_df, self.chrt_df, self.subj_bools_chrt, self.subj_bools_data,
          subproj_bools__data, and the outlier bools if they exist
        """

        reject_visits = self.qc_reject_ser.index[self.qc_reject_ser].tolist()
        orig_rej_cnt = len(reject_visits)
        self.qc_rejects_excluded = True     # flag for counts method

        # generate list of subjects to be rejected from subjid indexed df's
        reject_subjids = []
        for vid in reject_visits:
            sid = vid[0:8]
            if not sid in reject_subjids:
                reject_subjids.append(sid)

        if self.visit_code == "123":
            # also exclude other visits from same subject
            visits_to_add = []
            for vid in reject_visits:
                for suff in ["_01", "_02", "_03"]:
                    check_vid = vid[0:8] + suff
                    if (not check_vid in reject_visits) and (not check_vid in visits_to_add):
                        visits_to_add.append(check_vid)

            reject_visits.extend(visits_to_add)
            reject_visits.sort()

        print("Excluding {} visits (originally {} visits from {} subjects)...".format(len(reject_visits), orig_rej_cnt, len(reject_subjids)))
        print("Before exclusion, roi_df.shape: {}".format(self.roi_df.shape))

        # some df's are indexed by visit like the data
        if self.roi_gs_df is not None:
            vis_dfs = [self.roi_df, self.roi_gs_df] + self.subproj_bools__data
        else:
            vis_dfs = [self.roi_df] + self.subproj_bools__data

        for df in vis_dfs:
            df.drop(index=reject_visits, inplace=True)

        for (k, df) in self.subj_bools_data.items():
            # dictionaries with dict.items() create a list of (k,v) tuples
            df.drop(index=reject_visits, inplace=True)


        # other df's by subject like the cohort spreadsheets
        for df in [self.chrt_df]:
            df.drop(index=reject_subjids, inplace=True)

        for (k, df) in self.subj_bools_chrt.items():
            df.drop(index=reject_subjids, inplace=True)

        print("After exclusion, roi_df.shape: {}\n".format(self.roi_df.shape))


        # also exclude from the outlier dfs if they have been created
        if self.dist_outl_dfs is not None:
            for dodf in self.dist_outl_dfs:
                dodf.drop(index=reject_visits, inplace=True)

        if self.sse_outl_df is not None:
            self.sse_outl_df.drop(index=reject_subjids, inplace=True)


    def calc_subj_counts(self):
        """Count subjects in each group and associated outliers. Adds attribute
        self.subj_counts"""

        label_list = ["CTRL", "TRMT", "RESP", "NONR"]
        label_list.extend(RoiData.subprojs)

        count_list = [np.sum(self.subj_bools_chrt['ctrl']),
                      np.sum(~self.subj_bools_chrt['ctrl']),
                      np.sum(self.subj_bools_chrt['resp']),
                      np.sum(self.subj_bools_chrt['nonr'])]

        for sp_label in RoiData.subprojs:
            subproj_bool__chrt = self.chrt_df.Subproj == sp_label
            count_list.append(np.sum(subproj_bool__chrt))

        # subj_counts DF will later have a multi-index for columns
        subj_counts = pd.DataFrame(data={'count': count_list}, index=label_list)
        sc_columns = [['count'], ['']]


        # Outliers identified by QC
        # self.qc_reject_ser has one value per visit
        # this is True: np.all(self.subj_bools_data['ctrl'].index == self.qc_reject_ser.index)
        # so we can apply the data bools
        # Note 25 rejected visits in QC, and 8 were left out of TBSS analysis
        #   completely; there are 12 rejected subjs for '123' visit code, with
        #   the difference coming due to not having all 3 visits (MCU: 2, QNS: 1,
        #   UBC: 2)
        if self.qc_rejects_excluded:
            print("Skipping QC reject counts since they are already removed...")
        else:
            rej_ser = self.qc_reject_ser
            rej_count_list = [rej_ser[self.subj_bools_data['ctrl']].sum(),
                              rej_ser[~self.subj_bools_data['ctrl']].sum(),
                              rej_ser[self.subj_bools_data['resp']].sum(),
                              rej_ser[self.subj_bools_data['nonr']].sum()]

            for spbool in self.subproj_bools__data:
                rej_count_list.append(rej_ser[spbool].sum())

            # subj_counts['qc_reject'] = pd.Series(data=rej_count_list, index=label_list)
            subj_counts['qc_reject'] = rej_count_list
            sc_columns[0].extend(['qc_reject'])
            sc_columns[1].extend([''])


        # Outliers identified by SSE
        if self.sse_outl_df is None:
            print("Skipping SSE outlier counts since they aren't generated...")
        else:
            # self.sse_outl_df is by subject and ROI, not visit
            # this is True: np.all(self.subj_bools_chrt['ctrl'].index == self.sse_outl_df.index)
            # so we can apply the chrt bools directly
            # N.B. .values.sum() is necessary on the DFs to get total count for whole array
            #      otherwise .sum() operates per-column

            outl_df = self.sse_outl_df
            outl_count_list = []
            outl_min_list = []
            outl_max_list = []

            for gr_bool in [self.subj_bools_chrt['ctrl'],
                            ~self.subj_bools_chrt['ctrl'],
                            self.subj_bools_chrt['resp'],
                            self.subj_bools_chrt['nonr']]:
                outl_count_list.append(outl_df[gr_bool].values.sum())
                outl_min_list.append(outl_df[gr_bool].sum().min())
                outl_max_list.append(outl_df[gr_bool].sum().max())

            for sp_label in RoiData.subprojs:
                subproj_bool__chrt = self.chrt_df.Subproj == sp_label
                outl_count_list.append(outl_df[subproj_bool__chrt].values.sum())
                outl_min_list.append(outl_df[subproj_bool__chrt].sum().min())
                outl_max_list.append(outl_df[subproj_bool__chrt].sum().max())

            # normalize outlier counts by number of ROIs
            subj_counts['SSE_outl_mean'] = np.array(outl_count_list, dtype=float)/len(outl_df.columns)
            subj_counts['SSE_outl_min'] = np.array(outl_min_list, dtype=int)
            subj_counts['SSE_outl_max'] = np.array(outl_max_list, dtype=int)
            sc_columns[0].extend(['SSE_outl', 'SSE_outl', 'SSE_outl'])
            sc_columns[1].extend(['mean', 'min', 'max'])


        # Outliers identified by value distributions
        if self.dist_outl_dfs is None:
            print("Skipping dist outlier counts since they aren't generated...")
        else:
            # self.dist_outl_dfs[0] is by visit and ROI
            # this is True: np.all(self.subj_bools_data['ctrl'].index == self.dist_outl_dfs[0].index)
            # so we can apply the data bools

            # amalgamate the dfs in the list
            outl_df = np.full_like(self.dist_outl_dfs[0], False)
            for dodf in self.dist_outl_dfs:
                outl_df = outl_df | dodf

            outl_count_list = []
            outl_min_list = []
            outl_max_list = []

            for gr_bool in [self.subj_bools_data['ctrl'],
                            ~self.subj_bools_data['ctrl'],
                            self.subj_bools_data['resp'],
                            self.subj_bools_data['nonr']]:
                outl_count_list.append(outl_df[gr_bool].values.sum())
                outl_min_list.append(outl_df[gr_bool].sum().min())
                outl_max_list.append(outl_df[gr_bool].sum().max())

            for spbool in self.subproj_bools__data:
                outl_count_list.append(outl_df[spbool].values.sum())
                outl_min_list.append(outl_df[spbool].sum().min())
                outl_max_list.append(outl_df[spbool].sum().max())

            # normalize outlier counts by number of ROIs
            subj_counts['dist_outl_mean'] = np.array(outl_count_list, dtype=float)/len(outl_df.columns)
            subj_counts['dist_outl_min'] = np.array(outl_min_list, dtype=int)
            subj_counts['dist_outl_max'] = np.array(outl_max_list, dtype=int)
            sc_columns[0].extend(['dist_outl', 'dist_outl', 'dist_outl'])
            sc_columns[1].extend(['mean', 'min', 'max'])


        # Display counts with multi-index and add attribute
        subj_counts.columns = sc_columns
        with pd.option_context('display.float_format', '{:0.1f}'.format):
            print("Subject Counts:\n{}\n".format(subj_counts))

        self.subj_counts = subj_counts


    def calc_subj_sse(self, view=True, interactive=True, excl_outl_subjids=False):
        """Calculate the SSE of the linear regression of scalar values across the
        three time-points for each subject and ROI.

        Then calculates mean per subject across ROIs and mean per ROI across
        subjects.

        Uses the raw data (roi_df), since the global-scaling shouldn't really change CV/SSE.
        """

        assert (self.visit_code == "123"), "Must have 123 visit code for calculating SSE."
        assert (self.sse_outl_df is None), "Re-running calc_subj_sse is not supported."

        data = self.roi_df

        # Create df to hold subject/roi values considered outliers
        sse_outl_df = pd.DataFrame(data=False, index=self.chrt_df.index, columns=data.columns)

        # Get 1 linear fit for each subjid and ROI
        # Then can calculate mean SE per subj and mean per ROI
        se_dfs_bygroup = {}

        for group in ('ctrl', 'resp', 'nonr'):
            print("Calculating SE values for {}...".format(group))
            subj_bool = self.subj_bools_chrt[group]
            subjids = subj_bool.index[subj_bool].values

            # make dataframe to hold fit SE and intercept results
            #   size 91 x 43 for '123' data
            se_df = pd.DataFrame(index=subjids, columns=data.columns, dtype=float)
            int_df = pd.DataFrame(index=subjids, columns=data.columns, dtype=float)

            # note int_df is used later for plotting purposes, but not for
            #   detecting outliers

            regr = LinearRegression()
            for roi in data.columns:
                for sid in subjids:
                    a = data[roi][sid + '_01']
                    b = data[roi][sid + '_02']
                    c = data[roi][sid + '_03']

                    y = np.array([a, b, c]).reshape(-1, 1)

                    x = np.array([0.,2.,8.]).reshape(-1, 1)

                    # fit linear regression model for subject and calculate
                    #   residual SS (SSE) and standard error (SE) of the estimate
                    #   for SE def'n see Chernick2003, pp 263
                    regr.fit(x, y)
                    # regr.coef_       # array([[-0.00072746]])
                    # regr.intercept_  # array([0.78596654])
                    y_pred = regr.predict(x)
                    sse = np.sum((y - y_pred)**2)
                    se = np.sqrt(sse/1.)     # DF = 3 - 2

                    se_df.loc[sid, roi] = se
                    int_df.loc[sid, roi] = regr.intercept_[0]

                # Mark subject/rois with very high SE compared to the rest of the group
                # Using sqrt of SE to gain some normality (looks pretty good for most)
                outl_bool, zs_thr, _ = detect_outliers(np.sqrt(se_df[roi]), sided='high', report_thresh=False)
                # plot_hist_kde(np.sqrt(se_df[roi]), xlabel=group + " sqrt(SE) " + roi[0:20])

                outl_subjids = outl_bool.index[outl_bool].values
                sse_outl_df.loc[outl_subjids, roi] = True

                if np.any(outl_bool):
                    print("  High SSE outliers in {}: {}".format(abbrev_roi(roi), outl_subjids))

            print("Outlier z-score threshold for {} group was {:4.2f}".format(group, zs_thr))

            # still within group...
            # Plot mean SE values for each ROI
            roi_ses = se_df.mean(axis=0)
            roi_sestds = se_df.std(axis=0)

            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

            x = np.arange( len(roi_ses) ) + 1

            plt_kwargs={"ls": 'None', "marker": 'o', "mfc": 'None', "mec": (0., 0., 0., 0.5), "mew": 2, "label": None}
            # ax.plot(x, roi_ses, **plt_kwargs)
            ax.errorbar(x, roi_ses, yerr=roi_sestds, **plt_kwargs)

            # labels etc
            ax.set_xlabel("ROI")
            ax.set_ylabel("Regression SE ({} {})".format(group, self.metric))
            # ax.set_title("Raw SE for " + group)

            # draw, save, show fig
            fig.tight_layout()
            fig.canvas.draw()
            fig.savefig(self.roidata_dir + "ROI_SEs_" + group + "_" + self.metric + ".png", dpi=600)

            if view:
                fig.show()
            else:
                plt.close(fig)


            # mean SE values for each subject
            subj_ses = se_df.mean(axis=1)

            # Plot sqrt(SE) histogram for subjects
            #   Using sqrt of SE to gain some normality
            clrs = {'ctrl': ('orange',), 'resp': ('green',), 'nonr': ('blue',)}
            sse_hist, outl_ids = plot_hist_kde(np.sqrt(subj_ses), xlabel=r" $\sqrt{SE}$ for " + "{} {}".format(group, self.metric),
                                        clrs=clrs[group])
            sse_hist.savefig(self.roidata_dir + "Subj_SEs_" + group + "_" + self.metric + ".png", dpi=600)
            if view:
                sse_hist.show()
            else:
                plt.close(sse_hist)

            # Write SE values to CSV and add to dict
            se_df.to_csv(self.roidata_dir + "SE_" + group + "_subjids_x_rois-raw_" + self.metric + "_data.csv")
            se_dfs_bygroup[group] = se_df

            # roi_cvs.to_csv("cv_by_roi-raw_data.csv")            # mean 0.01440
            # subj_cvs.to_csv("cv_by_subjid-raw_data.csv")

            if interactive:
                vis_outl_str = raw_input("Enter no. of outlier subjects to examine (empty for none): ")
            else:
                vis_outl_str = ''

            try:
                vis_outl = int(vis_outl_str)
            except ValueError:
                if vis_outl_str == '':
                    vis_outl = int(0)
                else:
                    raise

            # Note the plots can help identify *possible* outliers -- then we must
            # look at those data and see if they should be excluded. Plot the linear
            # fits for the possible outlier subjects and check the dti image quality.

            # Plot the linear fits for the top outlier subjects
            if vis_outl > 0:
                subj_outl = subj_ses.sort_values(ascending=False)[0:vis_outl]
                print("Outliers to plot:\n{}\n".format(subj_outl))

                outdir = self.roidata_dir + "Subj_SE_outliers/"
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)

                for sid in subj_outl.index.values:
                    print("Examining " + sid)
                    # Examine high SE rois
                    subj_se = se_df.loc[sid]
                    outl_string = "Top outlier ROIs:\n{}".format(subj_se.sort_values(ascending=False)[0:10])
                    print(outl_string)
                    with open(outdir + "Subj_" + sid + "_" + self.metric + "_highest_SE.txt", mode='w') as of:
                        of.write(outl_string)

                    roi = raw_input("Enter ROI to plot (empty for none): ")

                    while roi != '':
                        a = data[roi][sid + '_01']
                        b = data[roi][sid + '_02']
                        c = data[roi][sid + '_03']

                        y = np.array([a, b, c]).reshape(-1, 1)

                        x = np.array([0.,2.,8.]).reshape(-1, 1)

                        # fit linear regression model for subject and calculate
                        #   residual SS
                        regr.fit(x, y)
                        # regr.coef_       # array([-0.00072746])
                        # regr.intercept_  # 0.7859665384615385
                        y_pred = regr.predict(x)
                        sse = np.sum((y - y_pred)**2)
                        se = np.sqrt(sse/1.)

                        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
                        plt_kwargs={"ls": 'None', "marker": 'o', "mfc": 'None', "mec": (0., 0., 0., 0.5), "mew": 2, "label": None}

                        ax.plot(x, y, **plt_kwargs)
                        ax.plot(x, y_pred, 'r-', label="SE: {se:01.2g}".format(se=se))

                        # also show mean and std of intercept for this ROI
                        v = int_df[roi].mean()
                        dv = int_df[roi].std()

                        plt_kwargs['marker'] = 'x'
                        plt_kwargs['label'] = "mean int. (sd={dv:01.2g})".format(dv=dv)
                        ax.errorbar(0, v, yerr=dv, **plt_kwargs)

                        # labels etc
                        ax.legend(loc='best')
                        ax.set_xlabel("Visit Week")
                        ax.set_ylabel(self.metric_label)
                        ax.set_title(sid + " " + short_name(roi))

                        fig.tight_layout()
                        fig.canvas.draw()
                        fig.savefig(outdir + "Subj_" + sid + "_" + roi + "_" + self.metric + "_regr.png", dpi=600)
                        if view:
                            fig.show()

                        roi = raw_input("Enter ROI to plot (empty for none): ")

                    print("")

            # Also plot the most typical subject for FA subj_ses: TGH_0079
            #   that subject's most typical roi is roi='Posterior_thalamic_radiation_include_optic_radiation_L'

            # sid='TGH_0079'
            # roi='Posterior_thalamic_radiation_include_optic_radiation_L'

            # a = data[roi][sid + '_01']
            # b = data[roi][sid + '_02']
            # c = data[roi][sid + '_03']

            # y = np.array([a, b, c]).reshape(-1, 1)

            # x = np.array([0.,2.,8.]).reshape(-1, 1)

            # # fit linear regression model for subject and calculate
            # #   residual SS
            # regr.fit(x, y)
            # # regr.coef_       # array([-0.00072746])
            # # regr.intercept_  # 0.7859665384615385
            # y_pred = regr.predict(x)
            # sse = np.sum((y - y_pred)**2)

            # fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            # plt_kwargs={"ls": 'None', "marker": 'o', "mfc": 'None', "mec": (0., 0., 0., 0.5), "mew": 2, "label": None}
            # ax.plot(x, y, **plt_kwargs)

            # ax.plot(x, y_pred, 'r-', label="SE: {sse:.1e}".format(sse=sse))

            # # labels etc
            # ax.legend(loc='best')
            # ax.set_xlabel("Visit Week")
            # ax.set_ylabel(self.metric)
            # ax.set_title("Median subject & ROI: " + sid + " " + roi)

            # fig.tight_layout()
            # fig.savefig(outdir + "Subj_median_fit_" + sid + "_" + roi + "_" + self.metric + "_regr.png", dpi=600)


            # In extreme subject cases, mark all ROIs as outliers
            if interactive:
                outl_subj_str = raw_input("Enter no. of subjects to exclude as outliers (empty for none): ")
            elif excl_outl_subjids:
                outl_subj_str = len(outl_ids)
            else:
                outl_subj_str = ''

            try:
                outl_subj_cnt = int(outl_subj_str)
            except ValueError:
                if outl_subj_str == '':
                    outl_subj_cnt = int(0)
                else:
                    raise

            if outl_subj_cnt > 0:
                subj_outls = subj_ses.sort_values(ascending=False)[0:outl_subj_cnt]

                for sid in subj_outls.index:
                    print("Marking {} as outlier for all rois.".format(sid))
                    sse_outl_df.loc[sid] = True     # scalar value sets all columns (rois) True


        print("Adding attributes: self.se_dfs_bygroup, self.sse_outl_df\n")
        self.se_dfs_bygroup = se_dfs_bygroup
        self.sse_outl_df = sse_outl_df


    def calc_gs_dists(self, view=True):
        """Plot histograms and KDEs for each ROI and find outliers. Uses roi_gs_df or
        imputed_roi_gs_df if it exists, creates dist_outl_dfs (or appends to it).
        """

        assert (self.roi_gs_df is not None), "Run remove_site_effects() first."

        # Set up output directory
        outl_dir = self.roidata_dir + "Dist-outliers_" + self.metric + "/"
        try:
            os.mkdir(outl_dir)
        except OSError:
            print("Warning: re-using existing outl_dir: {}".format(outl_dir))
            # print("Overwriting " + outl_dir) # don't do this for re-imputation of dist
            # shutil.rmtree(outl_dir)
            # os.mkdir(outl_dir)

        if self.imputed_roi_gs_df is None:
            data = self.roi_gs_df
        else:
            print("Using imputed_roi_gs_df (re-assessing outliers)")
            data = self.imputed_roi_gs_df

        # Plot histograms per ROI to check distributions for outliers
        # Note ROI with the biggest spread is probably 'Fornix_column_and_body_of_fornix'
        #   and the low outliers in that group are 'UBC_0068' 1/2/3 and 'UCA_0022' 1/2/3

        # Start with a fresh sesslist and dist-outliers
        outlier_sesslist = []
        dist_outl_df = pd.DataFrame(data=False, index=data.index, columns=data.columns)

        for roi in data.columns:
            print("Examining distribution in {} ...".format(roi))

            fig, outl_vis_ids = plot_hist_kde(data[roi], self.metric_label, title=short_name(roi),
                          groups=('ctrl', 'resp', 'nonr'), bools=self.subj_bools_data,
                          clrs=('orange', 'green', 'blue'))

            # note outliers for ROI
            with open(outl_dir + "outliers-{}.txt".format(roi), "w") as outfile:
                outfile.write("\n".join(outl_vis_ids) + "\n")

            outlier_sesslist.extend(outl_vis_ids)

            dist_outl_df.loc[outl_vis_ids, roi] = True

            fig.savefig(outl_dir + "dist-{}.png".format(roi), dpi=600)
            if view:
                # show figure and pause
                fig.show()
                raw_input("Press Enter to continue...")

            plt.close(fig)

        # record all outliers regardless of roi
        if len(outlier_sesslist) > 0:
            outlier_sesslist.sort()
            with open(outl_dir + "outlier_sessions_complete_list.txt", "a") as outfile:
                outfile.write("\n".join(outlier_sesslist) + "\n")

        # record outliers in attribute, adding to list if it exists to avoid re-imputing same values
        print("Adding attribute: dist_outl_dfs")
        if self.dist_outl_dfs is None:
            self.dist_outl_dfs = []

        self.dist_outl_dfs.append(dist_outl_df)


    def impute_outliers(self, outl_type=None):
        """Replace outlier values with ROI mean for the group. This may be called from
        the plotting method or the stat test methods.

        Impute the 'dist' or 'SSE' outlier types. Run twice for both.

        Adds attribute: self.imputed_roi_gs_df or self.imputed_roi_df
        """

        # Requires the outlier dfs, the data df, the group bools
        orig_data = self.roi_gs_df
        if self.imputed_roi_gs_df is None:
            imputed_data = self.roi_gs_df.copy()
        else:
            print("Using imputed_roi_gs_df (re-imputing)")
            imputed_data = self.imputed_roi_gs_df


        # Determine which visits values to impute
        if outl_type == 'dist':
            impute_visits_df = self.dist_outl_dfs[-1]   # impute only the latest run of dist-outliers

            orig_imp_cnt = np.sum(impute_visits_df.values)

            if self.visit_code == "123":
                # also impute other visits from same subject
                for roi in impute_visits_df.columns:
                    impute_visits = impute_visits_df.index[impute_visits_df[roi]].tolist()
                    visits_to_add = []
                    for vid in impute_visits:
                        for suff in ["_01", "_02", "_03"]:
                            check_vid = vid[0:8] + suff
                            if (not check_vid in impute_visits) and (not check_vid in visits_to_add):
                                visits_to_add.append(check_vid)

                    impute_visits_df.loc[visits_to_add, roi] = True

            print("Imputing {} dist outlier values (originally {} values)...".format(np.sum(impute_visits_df.values), orig_imp_cnt))

        elif outl_type == 'SSE':
            impute_subjids_df = self.sse_outl_df

            # translate subjids into visit ids and make a df with the full contingent
            #   of visit IDs
            impute_visits_df = pd.DataFrame(data=False, index=orig_data.index, columns=orig_data.columns)

            for roi in impute_visits_df.columns:
                impute_subjids = impute_subjids_df.index[impute_subjids_df[roi]].tolist()
                impute_visits = []
                for sid in impute_subjids:
                    impute_visits.extend([sid + '_01', sid + '_02', sid + '_03'])

                impute_visits_df.loc[impute_visits, roi] = True

            print("Imputing {} SSE outlier values (originally {} subjids) ...\n".format(np.sum(impute_visits_df.values), np.sum(impute_subjids_df.values)))

        else:
            raise ValueError("Choose and outl_type: dist or SSE")

        # Check each subject for whether there are any values to impute
        for sid in self.chrt_df.index:
            # visits associated with the subject
            if self.visit_code == "123":
                visits_to_check = [sid + '_01', sid + '_02', sid + '_03']
            else:
                visits_to_check = [sid + '_01']

            # N.B. impute_visits_df has an index and columns matching the data df bool
            if np.any(impute_visits_df.loc[visits_to_check]):
                # some values need imputing for this subject
                # determine subject's group
                if self.subj_bools_chrt['ctrl'].loc[sid]:
                    grp_bool = self.subj_bools_data['ctrl']
                elif self.subj_bools_chrt['resp'].loc[sid]:
                    grp_bool = self.subj_bools_data['resp']
                elif self.subj_bools_chrt['nonr'].loc[sid]:
                    grp_bool = self.subj_bools_data['nonr']
                else:
                    raise ValueError("Unable to determine group")

                # check if each particular ROI's values need imputing
                for roi in impute_visits_df.columns:
                    if np.all(impute_visits_df.loc[visits_to_check, roi]):
                        # impute each visit value in turn
                        for vid in visits_to_check:
                            vis_bool = impute_visits_df.index.str.endswith(vid[-3:])
                            grp_vis_bool = vis_bool & grp_bool
                            grp_vis_mean = orig_data.loc[grp_vis_bool, roi].mean()

                            imputed_data.loc[vid, roi] = grp_vis_mean

                    elif np.any(impute_visits_df.loc[visits_to_check, roi]):
                        # I don't think this should happen
                        raise ValueError("Inconsistency in impute_visits_df")


        # Record imputed data in attribute
        print("Adding attribute: imputed_roi_gs_df")
        self.imputed_roi_gs_df = imputed_data


    def regr_model_sa(self, ignore_SSE_outliers=True, view=True):
        """Regress sex and age variables from each subproject's CTRL v01 skeleton
        mean data to determine the (mean) dependence on age and sex.

        Write the fitted linear regression models for each metric as numpy objects
        to the v01 group_results dir.
        """

        # Extract skeleton mean metric data for v01
        skelmean01_ser = pd.read_csv(self.skelmean01_csv,
                                     header=None,
                                     names=["SUBJLABEL", "meandata"],
                                     index_col=0,
                                     squeeze=True)

        # Edit index to omit _01 visit code and match cohort df
        skelmean01_ser.index = skelmean01_ser.index.str[:8]

        ctrl01_bool__chrt = RoiData.chrt01_df.Group == "Control"  # 304x1 boolean array for v01

        # Centre age and sex
        ages = RoiData.chrt01_df.AGE - RoiData.chrt01_df.AGE.mean()   # 304 elem array matching cohort
        sexs = RoiData.chrt01_df.SEX - RoiData.chrt01_df.SEX.mean()

        # Find SSE outliers for skel-mean data
        if ignore_SSE_outliers:
            assert (self.sse_outl_df is not None), "run calc_subj_sse to use ignore_SSE_outliers"
            sse_outl_subjs = self.sse_outl_df.index[self.sse_outl_df.Skel_mean].tolist()
        else:
            sse_outl_subjs = None

        # Set up a plot, get list of matplotlib plot colours
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[6.5, 4.5], dpi=150)
        ax1.set_xlabel("Age")
        ax1.set_ylabel(self.metric_label)
        ax2.set_xlabel("Sex")
        ax2.set_ylabel(self.metric_label)
        # plt_clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

        plot_dir = self.roidata_dir + "regr_model_sa_plots/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)


        def subproj_regress(sp_label):
            """Regress sex and age for particular subproject.

            Return regression coefficients.
            """

            # Select controls from relevant subproject
            subproj01_bool__chrt = RoiData.chrt01_df.Subproj == sp_label
            faction_bool = ctrl01_bool__chrt & subproj01_bool__chrt

            N = sum(faction_bool)
            print("Regressing {:d} control subjects in subproj {} ...".format(N, sp_label))


            # Ensure the same subject IDs are being indexed in the cohort and data
            assert np.all(RoiData.chrt01_df.index.values[faction_bool] == skelmean01_ser.index.values[faction_bool]), "Index problem in subproj_regress."


            # Define vector data to regress
            y = skelmean01_ser[faction_bool]

            a = ages[faction_bool]
            s = sexs[faction_bool]

            # Drop SSE outliers if possible
            if sse_outl_subjs is not None:
                for sid in sse_outl_subjs:
                    if sid in y.index:
                        print("Ignoring SSE outlier: {}".format(sid))
                        y.drop(index=sid, inplace=True)
                        a.drop(index=sid, inplace=True)
                        s.drop(index=sid, inplace=True)


            # Do the linaer regression of sex and age on the data
            X = np.column_stack((a,s))

            regr = LinearRegression()
            regr.fit(X, y)
            # UCA_B:
            # regr.coef_        # array([0.00051606, 0.01141827])
            # regr.intercept_   # 0.5694800655312273
            r2 = regr.score(X, y)

            print("Regression results:")
            print("{:>3s}\t{:>8s}\t{:>7s}\t{:>6s}\t{:>6s}".format('N', 'B_age', 'B_sex', 'int', 'R^2'))
            print("{:3d}\t{: 0.5f}\t{: 0.4f}\t{: 0.3f}\t{: 0.3f}\n".format(N, regr.coef_[0], regr.coef_[1], regr.intercept_, r2))


            # Plot coeffs per subproj
            sk = np.argsort(a)      # sortkey
            ax1.plot(a[sk], y[sk], 'o')

            yhat_age = regr.intercept_ + regr.coef_[0]*a + regr.coef_[1]*s.mean()
            ax1.plot(a[sk], yhat_age[sk], 'k-')

            sk = np.argsort(s)      # sortkey
            ax2.plot(s[sk], y[sk], 'o')

            yhat_sex = regr.intercept_ + regr.coef_[0]*a.mean() + regr.coef_[1]*s
            ax2.plot(s[sk], yhat_sex[sk], 'k-')

            return regr, N, r2

        regrs_sa = []
        Ns = []
        r2s = []
        for sp_label in RoiData.subprojs:
            # print ("Regressing age and sex in {} ...".format(sp_label))
            regr, N, r2 = subproj_regress(sp_label)
            regrs_sa.append(regr)
            Ns.append(N)
            r2s.append(r2)


        # Make regression coefficients an attribute of the object
        self.regrs_sa = regrs_sa
        self.regr_sa_Ns = Ns
        self.regr_sa_r2s = r2s


        # Save and view the plots
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(plot_dir + "subprojs_regr_sa_" + self.metric + self.visit_code + ".png", dpi=600)

        if view:
            fig.show()
        else:
            plt.close(fig)


    def remove_sa_effects(self):
        """Remove sex and age effects from ROI data based on earlier regression.

        Adds attribute self.roi_saresid_df with residual ROI data.
        """

        assert (self.regrs_sa is not None), "Run regr_model_sa() first."

        # For the relevant cohort to this instance's data
        ages = self.chrt_df.AGE
        sexs = self.chrt_df.SEX

        # centre age and sex, the same way as in v01
        ages = ages - RoiData.chrt01_df.AGE.mean()
        sexs = sexs - RoiData.chrt01_df.SEX.mean()

        # Copy ROI data for processing, including JHU, skelmean, and skelperiph ROIs
        roi_saresid_df = self.roi_df.copy()

        def subt_model_params(y):
            """Returns data corrected for age and sex."""
            return y - b1*a - b2*s

        # Operate on data per subproject, including QNS_G and UBC_G
        for i, sp_label in enumerate(RoiData.subprojs):
            subproj_bool = self.chrt_df.Subproj == sp_label
            subjids_in_subproj = self.chrt_df.index[subproj_bool]
            ages_in_subproj = ages[subproj_bool]
            sexs_in_subproj = sexs[subproj_bool]

            # get sex/age regression coeffs for subproj, previously calculated from v01 data
            regr = self.regrs_sa[i]
            #b0 = regr.intercept_
            b1, b2 = regr.coef_     # age, sex regression coefficients

            for sid, a, s in zip(subjids_in_subproj, ages_in_subproj, sexs_in_subproj):
                # Select correct data rows for subjid
                row_selector = roi_saresid_df.index.str.startswith(sid)
                assert np.any(row_selector)

                # adjust values
                roi_saresid_df[row_selector] = subt_model_params(roi_saresid_df[row_selector])


        # Check for unmodified (missed) values
        assert not np.any(roi_saresid_df.values == self.roi_df.values)

        # Make residual data an attribute of the object
        print("Adding attribute: roi_saresid_df")
        self.roi_saresid_df = roi_saresid_df


    def regr_model_site_effects(self, ignore_SSE_outliers=True, view=True):
        """Calculate regression of each subproj's ROI mean residual values onto
        the global means. Uses CTRL v01 saresid data.

        Writes out regression coefficients and R^2 values for the site effect
        regressions.
        """

        assert (self.roi_saresid_df is not None), "Run remove_sa_effects() first."

        # Use visit 01 saresid data, without Skel_mean value
        data = self.roi_saresid_df.copy()
        data = data[data.index.str.endswith('_01')]
        data = data.drop(columns=self.dropped_rois)     # Skel_mean and Fornix

        # Drop SSE outliers if possible
        if ignore_SSE_outliers:
            assert (self.sse_outl_df is not None), "run calc_subj_sse to use ignore_SSE_outliers"
            outl_df = self.sse_outl_df.copy()
            outl_df = outl_df.drop(columns=self.dropped_rois)
            outl_df.index = [x + '_01' for x in outl_df.index]

            data[outl_df] = np.nan


        # Get mean ROI values for each subproj
        subproj_means_list = []     # will have 11 values after appends below
        for i, sp_label in enumerate(RoiData.subprojs):
            print("Calculating CTRL saresid ROI means for {} ...".format(sp_label))

            # Calculate mean ROI saresid values for ctrls in subproj
            subproj_ctrl_bool = self.subj_bools_data['ctrl'] & self.subproj_bools__data[i]
            subproj_ctrl_bool = subproj_ctrl_bool[subproj_ctrl_bool.index.str.endswith('_01')]

            subproj_means = data[subproj_ctrl_bool].mean()   # 42 elem dataseries

            # Print example values and append subproj means to list
            N = np.sum(subproj_ctrl_bool)
            eg_means = np.array_str(subproj_means.iloc[0:3].values, precision=3)
            print("  Example CC means for {:d} visit IDs: {}".format(N, eg_means))

            subproj_means_list.append(subproj_means)

        print("")

        # Calculate global ctrl mean values, make it a column
        global_ctrl_bool = self.subj_bools_data['ctrl']
        global_ctrl_bool = global_ctrl_bool[global_ctrl_bool.index.str.endswith('_01')]

        ctrl_means = data[global_ctrl_bool].mean()  # 42 elem series

        Ybar = ctrl_means.values.reshape(-1, 1)     # 42 x 1


        # Set up a plot
        fig, ax = plt.subplots(figsize=[6.5, 3], dpi=150)
        ax.set_xlabel("Global mean CTRL residual " + self.metric_label)
        ax.set_ylabel("Subproj mean CTRL residual")

        plot_dir = self.roidata_dir + "regr_model_site_plots/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)


        # Calculate linear regressors of subproj means onto global means
        regrs_site_effects = []
        r2s = []
        for i, sp_label in enumerate(RoiData.subprojs):
            # check for empty subprojects (e.g. UBC_A in 123 cohort)
            if np.any(np.isnan(subproj_means_list[i])):
                regrs_site_effects.append(np.nan)
                r2s.append(np.nan)
                continue

            # get values for subproj (i) and regress against global means
            Yibar = subproj_means_list[i].values

            regr = LinearRegression()
            regr.fit(Ybar, Yibar)
            # regr.coef_                    # array([[1.01814228]])
            # regr.intercept_               # array([0.60454502])
            r2 = regr.score(Ybar, Yibar)    # 0.982346927357766


            # Report subproj results and append to list
            print("Regression results for {}:".format(sp_label))
            print("{:>6s}  {:>6s}  {:>5s}".format('slope', 'int', 'R^2'))
            print("{: 0.3f}  {: 0.3f}  {:0.3f}\n".format(regr.coef_[0], regr.intercept_, r2))

            regrs_site_effects.append(regr)
            r2s.append(r2)


            # Plot regression results
            sk = np.argsort(Ybar.ravel())      # sortkey
            ax.plot(Ybar[sk], Yibar[sk], 'o')

            yhat = regr.intercept_ + regr.coef_[0]*Ybar
            ax.plot(Ybar[sk], yhat[sk], 'k-', alpha=0.75)


        # Add attributes
        print("Adding attributes: regrs_site_effects, regr_siteeffects_r2")
        self.regrs_site_effects = regrs_site_effects
        self.regr_siteeffects_r2 = r2s


        # Save and view the plots
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(plot_dir + "subprojs_regr_site_" + self.metric + self.visit_code + ".png", dpi=600)

        if view:
            fig.show()
        else:
            plt.close(fig)


    def remove_site_effects(self):
        """Adjust roi_df values based on site-effect regression coefficients,
        generating Global Scaled data. This generates self.roi_gs_df.
        """

        assert (self.regrs_site_effects is not None), "Run regr_model_site_effects() first."

        roi_gs_df = self.roi_df.copy()

        def apply_model(y):
            """Returns data corrected for site effects."""
            return (y - b0)/b1

        # Apply appropriate correction for each subproj
        for i, sp_label in enumerate(RoiData.subprojs):
            # check for empty subprojects (e.g. UBC_A in 123 cohort)
            try:
                if np.isnan(self.regrs_site_effects[i]):
                    print("{} was empty; skipping.\n".format(sp_label))
                    continue
            except TypeError:
                # regr is not a np.float type
                pass

            # load regressor and use it: y_gs = (y - b0)/b1
            regr = self.regrs_site_effects[i]
            b0 = regr.intercept_
            b1 = regr.coef_[0]

            print("applying (y-b0)/b1 for {}:".format(sp_label))
            print("N = {:d}".format(sum(self.subproj_bools__data[i])))
            print("b0, b1: {:0.6f}, {:0.6f}\n".format(b0, b1))

            roi_gs_df[self.subproj_bools__data[i]] = apply_model(roi_gs_df[self.subproj_bools__data[i]])

        # Add an attribute for GS data
        print("Adding attribute: roi_gs_df")
        self.roi_gs_df = roi_gs_df


    def drop_rois(self, rois):
        """Flag an ROI that is deemed unworthy. This will cause it to be ignored
        in plot_raw_gs(), calc_icc(), test_grpdiff_bsln(), and test_respdiff_wk8(), and
        model_lmer().

        Single argument can be a single roi or list. The value(s) will be added
        to the self.dropped_rois list."""

        self.dropped_rois.extend(rois)


    def plot_raw_gs(self, qc_excl='mark', sse_outl='mark', dist_outl='mark',
                    drop_rois=True, view=True, use_imputed=True,
                    fn_suffix=''):
        """Plot all ROI values against mean ROI value. The outliers options
        control behaviour on plots, and can be 'ignore', 'mark', or 'drop'. When
        the QC data points have been excluded already, the option is ignored. use_imputed
        can take the values True, False, and 'GS', which only uses imputed data for the
        GS corrected side of the plot.
        """

        assert (self.roi_gs_df is not None), "Run remove_site_effects() first."

        def plt_roi_data(df, ax, sortkey=None):
            """Generate consistent plots for raw and gs dataframes"""

            # Drop unwanted ROIs
            if drop_rois:
                df = df.drop(columns=self.dropped_rois)

            if sortkey is None:
                # Sort rois by mean ctrl value
                roi_cmeans = df[self.subj_bools_data['ctrl']].mean()
                sortkey = np.argsort(roi_cmeans.values)

            # means_tile = np.tile(roi_cmeans, [N, 1])  # create N copies, vertically stacked
            # y = df - means_tile
            # y_srt = y.apply(lambda row: row.iloc[sortkey], axis=1)
            y_srt = df.apply(lambda row: row.iloc[sortkey], axis=1)

            # create jittered count to plot against
            N = len(df.index)
            x0 = np.arange(df.shape[1], dtype='float')
            x = plt_jitter(x0, N)

            if qc_excl in ['mark', 'drop']:
                # check bad data identified in preliminary QC and either mark
                #   them with a blue 'o' or drop them from the plots
                #   this eliminates whole _visits_ (all rois), i.e. rows of df
                if self.qc_rejects_excluded:
                    print("Ignoring qc_rejects since they are already removed")
                else:
                    tf = self.qc_reject_ser

                    if qc_excl == 'mark':
                        ax.plot(x[tf], y_srt[tf], 'o', ms=5, mfc='None',
                            mec=(0,0,0.85,0.6), mew=1.5, zorder=0.5)
                    elif qc_excl == 'drop':
                        y_srt[tf] = np.nan

            if sse_outl in ['mark', 'drop']:
                # also check each roi for outliers identified by SSE
                #   and mark them with a red 'x'
                if self.sse_outl_df is None:
                    print("Ignoring SSE outliers as they have not been identified")
                else:
                    for roi in y_srt.columns:
                        tf = self.sse_outl_df[roi]
                        if np.any(tf):
                            # get matrix indices of outlier subjids to match with x
                            outl_subjids = tf.index.values[tf]

                            for sid in outl_subjids:
                                for vid in ('_01', '_02', '_03'):
                                    row = y_srt.index.get_loc(sid + vid)
                                    col = y_srt.columns.get_loc(roi)

                                    if sse_outl == 'mark':
                                        ax.plot(x[row, col], y_srt.iloc[row, col], 'x',
                                                ms=4., mec=(0.85,0,0,0.6), zorder=1)
                                    elif sse_outl == 'drop':
                                        y_srt.iloc[row, col] = np.nan

            if dist_outl in ['mark', 'drop']:
                # also check for outliers identified by data distribution
                #   and mark them with a red '+'
                #   this matrix index matches y
                if self.dist_outl_dfs is None:
                    print("Ignoring dist outliers as they have not been identified")
                else:
                    tf = np.full_like(self.dist_outl_dfs[0], False)
                    for dodf in self.dist_outl_dfs:
                        tf = tf | dodf

                    if np.any(tf):
                        tf_srt = tf.apply(lambda row: row.iloc[sortkey], axis=1)

                        if dist_outl == 'mark':
                            ax.plot(x[tf_srt], y_srt.values[tf_srt], '+',
                                    ms=5., mec=(0.85,0,0,0.6), zorder=1)
                        elif dist_outl == 'drop':
                            y_srt.values[tf_srt] = np.nan


            # Plot ROI data against arange (not de-meaned anymore)
            plt_kwargs={'ls': 'None', 'marker': 'o', 'ms': 1.5, 'mfc': 'None',
                        'mec': (0.,0.,0.,0.4), 'mew': 0.9,
                        'zorder': 3, 'label': None}
            # check defaults with e.g. plt.rcParams['lines.markersize']
            # plt_kwargs1={"ls": 'None', "marker": 'o', "mfc": 'None', "mec": (0.5,0.,0.,0.5), "mew": 2, "label": None}
            # plt_kwargs2={"ls": 'None', "marker": 'o', "mfc": 'None', "mec": (0.,0.,0.5,0.5), "mew": 2, "label": None}

            ax.plot(x, y_srt, **plt_kwargs)


            # plot subproj means as coloured lines
            for i, tf in enumerate(self.subproj_bools__data):
                subproj_means = df[tf].mean()
                # sm_srt = subproj_means[sortkey] - roi_cmeans[sortkey]
                sm_srt = subproj_means[sortkey]
                ax.plot(x0, sm_srt, '-', lw=1.0, alpha=0.85, zorder=5, label=RoiData.subprojs[i])

            ax.set_xlim([-1, 41])
            # ax.set_ylim([-0.42, 0.42])

            return sortkey, df.columns[sortkey].tolist()     # list of ROIs in order in the plots

        # Create figure and set up plot options
        #   use markers for data points, lines for project means
        plt.rcParams['font.size'] = 9.
        plt.rcParams['axes.titlepad'] = 3.0
        plt.rcParams['axes.labelpad'] = 2.0
        plt.rcParams['ytick.major.pad'] = 1.5
        plt.rcParams['xtick.major.pad'] = 2.0
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.48, 2.4), dpi=150, sharey=True)

        # Use custom sortkey based on increasing CTRL mean FA with manual edits to keep pairs together
        sortkey = np.array([23, 24, 39, 40, 13, 14, 15, 16, 17, 18, 35, 36, 31, 32, 33, 34, 21, 22, 29, 30, 27, 28, 11, 12, 7, 8, 37, 38, 25, 26, 19, 20, 3, 4, 9, 10, 5, 6, 0, 1, 2])

        # Plot Raw data on ax1
        if (use_imputed == True) and (self.imputed_roi_df is None):
            print("imputed_roi_df not available; using roi_df")
            _, _ = plt_roi_data(self.roi_df, ax1, sortkey=sortkey)
        elif (use_imputed == True):
            _, _ = plt_roi_data(self.imputed_roi_df, ax1, sortkey=sortkey)
        else:
            _, _ = plt_roi_data(self.roi_df, ax1, sortkey=sortkey)

        # ylabel
        if self.visit_code == '1':
            ax1.set_ylabel(self.metric_label + " (baseline visit)")
        elif self.visit_code == '123':
            ax1.set_ylabel(self.metric_label)

        # Plot GS data on ax2
        # roi_gs_deltas_df = plt_roi_data(self.roi_gs_df, ax2)
        if ((use_imputed == True) or (use_imputed == 'GS')) and (self.imputed_roi_gs_df is None):
            print("imputed_roi_gs_df not available; plotting roi_gs_df")
            _, rois_sorted = plt_roi_data(self.roi_gs_df, ax2, sortkey=sortkey)
        elif ((use_imputed == True) or (use_imputed == 'GS')):
            _, rois_sorted = plt_roi_data(self.imputed_roi_gs_df, ax2, sortkey=sortkey)
        else:
            _, rois_sorted = plt_roi_data(self.roi_gs_df, ax2, sortkey=sortkey)

        if self.metric == 'FA':
            # in final figure, only FA needs titles
            ax1.set_title("Raw")
            ax2.set_title("Global Scaled")

            ax1.tick_params(labelbottom=False)
            ax2.tick_params(labelbottom=False)
        elif self.metric == 'RD':
            # only RD needs x-axis labels
            ax1.set_xlabel("ROI")
            ax2.set_xlabel("ROI")
        else:
            ax1.tick_params(labelbottom=False)
            ax2.tick_params(labelbottom=False)

        fig.tight_layout()

        # set subplots_adjust to get consistent placements that work for all metrics
        # interrogate values with e.g. fig = obj.fig_raw_gs; fig.subplotpars.left
        fig.subplots_adjust(left=0.06, bottom=0.135, right=0.99, top=0.93, wspace=0.035, hspace=None)

        fig.canvas.draw()
        fig.savefig(self.roidata_dir + "ROI_subproj_" + self.metric + "_raw+gs" + fn_suffix + ".png", dpi=1000)
        with open(self.roidata_dir + "ROI_subproj_" + self.metric + "_sortkey" + fn_suffix + ".txt", 'w') as of:
            of.write('\n'.join(rois_sorted))

        if view:
            print("\nSorted ROIs:\n"
                + "\n".join(rois_sorted) + "\n")
            fig.show()
        else:
            plt.close(fig)

        # Note: 6 outliers in Fornix_column_and_body_of_fornix, identified with:
        #np.sum(raw_delta < -0.26)
        #raw_delta['Fornix_column_and_body_of_fornix'][raw_delta['Fornix_column_and_body_of_fornix'] < -0.26]
        # result:
        # UBC_0068_01   -0.308085
        # UBC_0068_02   -0.294448
        # UBC_0068_03   -0.279142
        # UCA_0022_01   -0.350384
        # UCA_0022_02   -0.334210
        # UCA_0022_03   -0.324291

        # Also 1 outlier in Superior_fronto-occipital_fasciculus_could_be_a_part_of_anterior_internal_capsule_R:
        # UBC_0074_01    0.339523

        # all values for this subject:
        # UBC_0074_01    0.339523
        # UBC_0074_02    0.008990
        # UBC_0074_03    0.168049

        # add figure as an attribute
        self.fig_raw_gs = fig


    def calc_icc(self, drop_rois=True, view=True):
        """Calculate ICC values across raw and GS ROI data. Does not use imputed data.

        Uses the nipype implementation of ICC(3,1):
        http://nipype.readthedocs.io/en/latest/interfaces/generated/nipype.algorithms.icc.html
        """

        assert (self.roi_gs_df is not None), "Run remove_site_effects() first."
        assert (self.visit_code == "123"), "Must be using 123 visit code for ICC."

        def calc_roi_iccs(data):
            """Calculate ICC values per ROI for the passed dataframe (raw, GS, etc.)"""

            if drop_rois:
                print("Ignoring dropped_rois: {}".format(self.dropped_rois))
                data = data.drop(columns=self.dropped_rois)

            # make dataframe to hold results
            m = len(data.columns)
            icc_df = pd.DataFrame(data=np.zeros((m,4)), index=data.columns, columns=['icc', 'subj_var', 'sess_var', 'sess_F'])

            for roi in data.columns:
                # Reshape each ROI array to 3 columns for visits and make it a np array
                # from the ICC_rep_anova(Y) code: [nb_subjects, nb_conditions] = Y.shape
                roivals_array = data[roi].values.reshape([-1, 3])

                iccval, subj_var, sess_var, sess_F, _, _ = icc.ICC_rep_anova(roivals_array)
                icc_df.loc[roi] = [iccval, subj_var, sess_var, sess_F]

            return icc_df

        # Calc ICC on raw and gs data
        icc_raw_df = calc_roi_iccs(self.roi_df)
        icc_raw_df.to_csv(self.roidata_dir + "icc_" + self.metric + "_rois-raw_data.csv")   # icc_raw_df['icc'].mean(): 0.853

        icc_gs_df = calc_roi_iccs(self.roi_gs_df)
        icc_gs_df.to_csv(self.roidata_dir + "icc_" + self.metric + "_rois-gs_data.csv")    # icc_gs_df['icc'].mean(): 0.836

        print("Adding attributes: self.icc_gs_df, self.icc_raw_df")
        self.icc_gs_df = icc_gs_df


        # Plot GS ICCs per ROI
        # Is plot order the same as plot_raw_gs?
        fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
        m = icc_gs_df.shape[0]
        x = np.arange(m) + 1

        plt_kwargs={"ls": 'None', "marker": 'o', "mfc": 'None', "mec": (0., 0., 0., 0.5), "mew": 2, "label": None}
        ax.plot(x, icc_gs_df['icc'], **plt_kwargs)

        # labels etc
        ax.set_xlabel("ROI")
        ax.set_ylabel("ICC")
        ax.set_title("GS ICC Values")

        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(self.roidata_dir + "ROI_ICC_" + self.metric + ".png", dpi=600)

        if view:
            fig.show()
        else:
            plt.close(fig)


    def test_grpdiff_bsln(self):
        """Test for group differences in baseline data using t-test or ANOVA. Uses roi_gs_df
        data, or imputed_roi_gs_df if available.  Adds grpdiff_bsln pandas df as
        attribute, and writes out csv: groupdiffs-baseline_XX.csv.
        """

        assert (self.roi_gs_df is not None), "Run remove_site_effects() first"

        # Define log file
        logfile = self.roidata_dir + "grpdiff_bsln_" + self.metric + "_log.txt"
        if os.path.isfile(logfile):
            os.remove(logfile)

        # Use data with outliers excluded/imputed if available
        if self.imputed_roi_gs_df is None:
            print_log("Imputed data not available, using roi_gs_df", logfile)
            data = self.roi_gs_df.copy()
        else:
            data = self.imputed_roi_gs_df.copy()

        # Ignore skel_mean value, other ROIs deemed unworthy
        data = data.drop(columns=self.dropped_rois)

        # Define booleans to select baseline values and combine with ctrl/trmt/resp/nonr
        bsln_bool = data.index.str.endswith('_01')

        ctrl_bl = bsln_bool & self.subj_bools_data['ctrl']      # 103 at baseline
        trmt_bl = bsln_bool & ~self.subj_bools_data['ctrl']     # 188
        resp_bl = bsln_bool & self.subj_bools_data['resp']      # 80
        nonr_bl = bsln_bool & self.subj_bools_data['nonr']      # 85


        # Create dataframe to hold results -- rois x t,p,pcorr,reject,reject_soft
        column_labs=[['trmt-ctrl']*5 + ['trigroup_AOV']*5 + ['resp-ctrl']*2 + ['nonr-ctrl']*2 + ['nonr-resp']*2,
                     ['t','p','pcorr','rej_H0','rej_soft'] + ['F','p','pcorr','rej_H0','rej_soft'] + ['rej_H0', 'meandiff']*3]
        stats_df = pd.DataFrame(index=data.columns, columns=column_labs)

        # Run group comparisons in each ROI
        #   using t-test for MDD-CTRL and
        #   ANOVA for trigroup comparisons
        for roi in data.columns:
            roi_ser = data[roi]

            # T-test for MDD-CTRL and populate the dataframe with raw p-values
            #   N.B. variances seem very close to equal
            t, p = stats.ttest_ind(roi_ser[trmt_bl], roi_ser[ctrl_bl])

            stats_df.loc[roi, ('trmt-ctrl', 't')] = t
            stats_df.loc[roi, ('trmt-ctrl', 'p')] = p


            # ANOVA for trigroup comparison
            F, p = stats.f_oneway(roi_ser[ctrl_bl], roi_ser[resp_bl], roi_ser[nonr_bl])

            stats_df.loc[roi, ('trigroup_AOV', 'F')] = F
            stats_df.loc[roi, ('trigroup_AOV', 'p')] = p


        # Control for multiple comparisons with FDR for both sets of p-values
        stats_df = stats_df.infer_objects()
        for test_grps in ['trmt-ctrl', 'trigroup_AOV']:
            rej_H0, pcorr, _, _ = sm.stats.multipletests(stats_df[(test_grps, 'p')].values,
                                                         alpha=0.05,
                                                         method='fdr_tsbh')
            rej_soft = pcorr < 0.10

            stats_df[(test_grps, 'pcorr')] = pcorr
            stats_df[(test_grps, 'rej_H0')] = rej_H0
            stats_df[(test_grps, 'rej_soft')] = rej_soft

            print_log("Baseline " + test_grps + ": Any Grp_FDR-p < 0.05? " + \
                      "{any} ({sum})".format(any=np.any(rej_H0), sum=np.sum(rej_H0)), logfile)

            if np.any(rej_H0):
                print_log(stats_df.loc[rej_soft, test_grps].to_string(), logfile)
                print_log("", logfile)


        # Create trigroup data column and drop subjects without the resp/nonr determination
        trigroup = pd.Series(data='', index=data.index, dtype='object')
        trigroup[ctrl_bl] = 'ctrl'
        trigroup[resp_bl] = 'resp'
        trigroup[nonr_bl] = 'nonr'

        drop_list = trigroup.index[trigroup=='']        # len = 23 at BL
        trigroup.drop(index=drop_list, inplace=True)
        data_tg = data.drop(index=drop_list)


        # Follow up with post-hoc Tukey HSD for significant trigroup effects in ANOVA
        for roi in stats_df.index[stats_df[('trigroup_AOV', 'rej_H0')]]:
            # Assume pairwise group tests in the DF are False unless HSD tests say otherwise
            stats_df.loc[roi, (['resp-ctrl', 'nonr-ctrl', 'nonr-resp'], 'rej_H0')] = False

            # HSD test: default alpha=0.05, returns SM results class
            tukeyhsd_result = sm.stats.multicomp.pairwise_tukeyhsd(data_tg[roi], trigroup)

            print_log("Tukey HSD test for {}:".format(roi), logfile)
            print_log(tukeyhsd_result.summary().as_text(), logfile)
            print_log("", logfile)

            # note order of HSD results table is N-C, R-C, R-N
            if tukeyhsd_result.reject[0]:
                stats_df.loc[roi, ('nonr-ctrl', 'rej_H0')] = True
                stats_df.loc[roi, ('nonr-ctrl', 'meandiff')] = tukeyhsd_result.meandiffs[0]

            if tukeyhsd_result.reject[1]:
                stats_df.loc[roi, ('resp-ctrl', 'rej_H0')] = True
                stats_df.loc[roi, ('resp-ctrl', 'meandiff')] = tukeyhsd_result.meandiffs[1]

            if tukeyhsd_result.reject[2]:
                stats_df.loc[roi, ('nonr-resp', 'rej_H0')] = True
                stats_df.loc[roi, ('nonr-resp', 'meandiff')] = tukeyhsd_result.meandiffs[2]


        # N.B. could have used a linear model, the statsmodels way
        # ph_lm = sm.formula.ols('fe_vals ~ trigroup', data=fixef_df).fit()
        # #ph_lm.summary()

        # aov_table = sm.stats.anova_lm(ph_lm, typ='II') # Type 2 ANOVA DataFrame
        # print(aov_table)

        # Use abbreviated ROI names
        idx_full = stats_df.index.tolist()
        idx_abrv = [abbrev_roi(s) for s in idx_full]
        stats_df.rename(index=dict(zip(idx_full, idx_abrv)), inplace=True)


        # Write out results and add attribute
        stats_df.to_csv(self.roidata_dir + "groupdiffs-baseline_" + self.metric + ".csv")

        print("Adding attribute: self.grpdiff_bsln")
        self.grpdiff_bsln = stats_df


    def test_respdiff_wk8(self):
        """Test group differences: resp vs non-resp at week 8. Uses roi_gs_df, or
        imputedroi_gs_df data if available.
        """

        assert (self.roi_gs_df is not None), "Run remove_site_effects() first."

        print("Warning: This method may not be so useful anymore...")

        # Use data with outliers excluded/imputed if available
        if self.imputed_roi_gs_df is None:
            print("Imputed data not available, using roi_gs_df")
            data = self.roi_gs_df.copy()
        else:
            data = self.imputed_roi_gs_df.copy()

        # Ignore skel_mean value, other ROIs deemed unworthy
        data = data.drop(columns=self.dropped_rois)

        # Define booleans to select wk8 values and combine with resp/nonr
        wk8_bool = data.index.str.endswith('_03')

        resp_wk8 = wk8_bool & self.subj_bools_data['resp']
        nonr_wk8 = wk8_bool & self.subj_bools_data['nonr']


        # Create dataframe to hold results -- rois x t,p
        m = len(data.columns)
        tp_df = pd.DataFrame(data=np.zeros([m,2]), index=data.columns, columns=['t', 'p'])

        for roi in data.columns:
            roi_ser = data[roi]

            # note variances seem very close to equal
            t, p = stats.ttest_ind(roi_ser[resp_wk8], roi_ser[nonr_wk8])

            tp_df.loc[roi] = [t, p]

        # control for multiple comparisons with FDR
        rej_H0, pcorr, _, _ = sm.stats.multipletests(tp_df['p'].values, alpha=0.05, method='fdr_tsbh')
        rej_soft = pcorr < 0.10

        tp_df['pcorr'] = pcorr
        tp_df['rej_H0'] = rej_H0
        tp_df['rej_soft'] = rej_soft

        print("Week 8 NONR/RESP: Any FDR-p < 0.05? ", end='')
        print("{any} ({sum})".format(any=np.any(rej_H0), sum=np.sum(rej_H0)))

        if np.any(rej_H0):
            print(tp_df.loc[tp_df.rej_soft == True])

        tp_df.to_csv(self.roidata_dir + "groupdiff-wk8_resp_vs_nonr_" + self.metric + ".csv")

        print("Adding attribute: self.groupdiff_wk8")
        self.groupdiff_wk8 = tp_df


    def model_lmer(self):
        """Run the linear mixed effects model on the GS ROI data"""

        # Goals of the modelling and hypothesis testing; main questions to answer:
        #   - is there a baseline difference between ctrl and trmt or resp/nonresp groups
        #   - is there a visit-time slope associated with the resp group?

        # Notes on model design:
        #   - Visit-time is numerical and our main regressor, so it's a fixed effect.
        #     This will produce a coefficient for the temporal slope of the response
        #     (e.g. the association btw FA and time).
        #   - The study groups are not part of an exchangeable set: therefore they
        #     represent a fixed effect in the model. Adding the trigroup parameter
        #     will generate an intercept for each group (ctrl/resp/nonr)
        #   - The interaction of study-groups and visit-time is a fixed-effect, since
        #     they're both fixed. This gives each group a unique time slope.
        #   - Subjects are essentially exchangeable within their set, so that's a random
        #     effect. In this model the effect implied by the intercept is value
        #     at time-0. Subjects should get a random slope as well.
        #
        # formula: roidata ~ 1 + visit_week + trigroup + trigroup:visit_week + (1 + visit_week|subjid)

        # Galwey 7.5 formula
        #   - note baseline value is subtracted from the data, and gets its own column, then
        #     the response variable is called "change"
        #   - change ~ baseline + treatment * visit_day + ((1 + visit_day)|id)
        #   - this is not so good for us, I think, because we are actually interested
        #     in the effect of baseline

        # Multiple ways to model this data:
        #   - independent models for each study group, compare fixed effects:
        #     F, p = scipy.stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])
        #     for post-hoc tests, can use t-tests, or pairwise_tukeyhsd (see http://hamelg.blogspot.ca/2015/11/python-for-data-analysis-part-16_23.html)
        #   - model all together -> use SE of BLUPs to figure out if Group BLUPs differ from 0
        #   - model with FE mean of ctrl group removed from trmt group, then look at probability of difference

        # Full model specification:
        # dependent/outcome variable:
        #   - roi metric values
        # fixed effects:
        #   - visit_week
        #   - study group
        # random effects:
        #   - subject id

        # note: tried removing the interaction term between RE slope and intercept
        #   abandoned that because LL was higher with it (p=0.07)

        # Notes:
        #   - try out the factors command for the pymer4 fit() method, see http://eshinjolly.com/pymer4/categorical.html#dummy-coding-factors
        #       see modl.post_hoc -> only tests 'factors', none listed in my output as no anova done
        #   - also take a look at the 'Inspect' parts there, at least for significant ROIs
        #   - to see all output methods and attributes from the lmer model object, see table 8 on pp 30 of https://arxiv.org/pdf/1406.5823.pdf

        assert (self.roi_gs_df is not None), "Run remove_site_effects() first."

        # Use data with outliers excluded/imputed if available
        if self.imputed_roi_gs_df is None:
            print("Imputed data not available, using roi_gs_df")
            data = self.roi_gs_df.copy()
        else:
            data = self.imputed_roi_gs_df.copy()

        # Ignore skel_mean value, other ROIs deemed unworthy
        data = data.drop(columns=self.dropped_rois)

        # Define log file
        logfile = self.roidata_dir + "lmer_model_" + self.metric + "_log.txt"
        if os.path.isfile(logfile):
            os.remove(logfile)

        # Create subject ID and visit time columns based on the index
        subjid_arr = data.index.str[0:8].values
        visit_week_arr = np.full(data.index.shape, np.nan)
        for visit_id, visit_week in zip(['_01', '_02', '_03'], [0., 2., 8.]):
            vis_bool = data.index.str.endswith(visit_id)
            visit_week_arr[vis_bool] = visit_week

        # Two groups: Control and Treatment
        twogroup_labels = []
        for tf in self.subj_bools_data['ctrl']:
            if tf:
                twogroup_labels.append('ctrl')
            else:
                twogroup_labels.append('trmt')

        twogroup_labels = pd.Series(data=twogroup_labels, index=data.index)

        # Three groups: Control, Responder, Nonresponder
        trigroup_labels = []
        for i, tf in self.subj_bools_data['ctrl'].iteritems():
            if tf:
                trigroup_labels.append('ctrl')
            else:
                if self.subj_bools_data['resp'].loc[i]:
                    trigroup_labels.append('resp')
                elif self.subj_bools_data['nonr'].loc[i]:
                    trigroup_labels.append('nonr')
                else:
                    raise ValueError("Unknown trigroup for visit {}".format(i))

        trigroup_labels = pd.Series(data=trigroup_labels, index=data.index)


        def print_modl_sig(modl, roi):
            """Check to see if model fixed effects were significant by F-test"""

            # The Sig column gets stars *** or . for p<0.1, empty for p>0.1
            for effect in modl.coefs.index:
                if effect == "(Intercept)":
                    # of course there is a non-zero intercept
                    continue
                if modl.coefs.loc[effect, 'Sig'] != '':
                    print_log("Significant effect of {effect} in {roi}: p={pval:01.4f}".format(
                          effect=effect, roi=roi, pval=modl.coefs.loc[effect, 'P-val']), logfile)

        # Create dataframe to hold p-values and dicts for the model objects
        lmer_pvals = pd.DataFrame(index=data.columns,
                                  columns=["LR_group", "LR_time", "LR_gxt",
                                           "2G_trmt", "2G_time", "2G_gxt",
                                           "3G_resp", "3G_nonr", "3G_time", "3G_rxt", "3G_nxt"],
                                  dtype=float)
        lmer_modls_2g = {}
        lmer_modls_3g = {}

        # Test mixed model on each ROI
        pool = mp.Pool(processes=3)     # for parallelism
        for roi in data.columns:
            print_log("Testing ROI: {roi}".format(roi=roi), logfile)
            # create df for each ROI's data and fit model to it
            single_roi_data = data[roi].to_frame()  # returns a copy
            single_roi_data.rename(columns={roi: 'roidata'}, inplace=True)
            single_roi_data['subjid'] = subjid_arr
            single_roi_data['visit_week'] = visit_week_arr

            # # for diffusion values, can be easier to see:
            # single_roi_data['roidata'] *= 1000 # to make MD values more manageable

            # Full models:
            #   2g including ctrl and treatment study groups
            #   3g including ctrl, resp, nonr groups
            single_roi_data['twogroup'] = twogroup_labels
            modl_2g = pm4dev.Lmer('roidata ~ 1 + twogroup + visit_week + twogroup:visit_week + (1 + visit_week|subjid)',
                                  data=single_roi_data)

            single_roi_data.drop(columns='twogroup', inplace=True)  # clean up to be safe

            single_roi_data['trigroup'] = trigroup_labels
            modl_3g = pm4dev.Lmer('roidata ~ 1 + trigroup + visit_week + trigroup:visit_week + (1 + visit_week|subjid)',
                                  data=single_roi_data)

            # do the fitting in parallel
            modl_2g, modl_3g = pool.map(run_modl_fit, [(modl_2g, logfile),
                                                       (modl_3g, logfile)])

            # check p-values and add to table (except intercept)
            print_modl_sig(modl_2g, roi)    # checks p-val of visit_week and twogrouptrmt
            lmer_pvals.loc[roi, '2G_trmt'] = modl_2g.coefs.loc['twogrouptrmt', 'P-val']
            lmer_pvals.loc[roi, '2G_time'] = modl_2g.coefs.loc['visit_week', 'P-val']
            lmer_pvals.loc[roi, '2G_gxt'] = modl_2g.coefs.loc['twogrouptrmt:visit_week', 'P-val']

            print_modl_sig(modl_3g, roi)
            lmer_pvals.loc[roi, '3G_resp'] = modl_3g.coefs.loc['trigroupresp', 'P-val']
            lmer_pvals.loc[roi, '3G_nonr'] = modl_3g.coefs.loc['trigroupnonr', 'P-val']
            lmer_pvals.loc[roi, '3G_time'] = modl_3g.coefs.loc['visit_week', 'P-val']
            lmer_pvals.loc[roi, '3G_rxt'] = modl_3g.coefs.loc['trigroupresp:visit_week', 'P-val']
            lmer_pvals.loc[roi, '3G_nxt'] = modl_3g.coefs.loc['trigroupnonr:visit_week', 'P-val']

            # report results of Tukey HSD
            old_stdout = sys.stdout     # redirecting stdout due to annoying message from post_hoc
            sys.stdout = open(os.devnull, 'w')

            tkhsd_2g = modl_2g.post_hoc('twogroup')
            tkhsd_3g = modl_3g.post_hoc('trigroup')

            sys.stdout = old_stdout

            if np.any(tkhsd_2g[1]['p.value'] < 0.1):
                print_log("Significant (maybe soft) result from twogroup Tukey HSD in {}:".format(roi), logfile)
                print_log(tkhsd_2g[0].to_string(), logfile)
                print_log(tkhsd_2g[1].to_string(), logfile)

            if np.any(tkhsd_3g[1]['p.value'] < 0.1):
                print_log("Significant (maybe soft) result from trigroup Tukey HSD in {}:".format(roi), logfile)
                print_log(tkhsd_3g[0].to_string(), logfile)
                print_log(tkhsd_3g[1].to_string(), logfile)

            # Populate dicts with these models to retrieve later
            #   each one is about 2.3 Mb in memory
            # if p_group_LR < 0.10:  # <- no, just save them all
            lmer_modls_2g[roi] = modl_2g
            lmer_modls_3g[roi] = modl_3g

            # Test significance of effects using likelihood ratio of models
            #   with and without the term of interest
            # Specify null and alt formulas, data, and difference in DFs
            # Running in parallel

            LR_arg_tuples = (('roidata ~ 1 + visit_week + (1 + visit_week|subjid)',
                              'roidata ~ 1 + trigroup + visit_week + (1 + visit_week|subjid)',
                              single_roi_data,
                              1,
                              logfile),
                             ('roidata ~ 1 + trigroup + (1 + visit_week|subjid)',
                              'roidata ~ 1 + trigroup + visit_week + (1 + visit_week|subjid)',
                              single_roi_data,
                              1,
                              logfile),
                             ('roidata ~ 1 + trigroup + visit_week + (1 + visit_week|subjid)',
                              'roidata ~ 1 + trigroup + visit_week + trigroup:visit_week + (1 + visit_week|subjid)',
                              single_roi_data,
                              1,
                              logfile))

            p_group_LR, p_time_LR, p_gxt_LR = pool.map(LLR_test, LR_arg_tuples)

            for pval, effect in zip((p_group_LR, p_time_LR, p_gxt_LR), ("trigroup", "time", "gxt")):
                if pval < 0.1:
                    print_log("Likelihood ratio indicates significant effect of {effect} in {roi}: p={pval:01.4f}".format(
                          effect=effect, roi=roi, pval=pval), logfile)

            lmer_pvals.loc[roi, 'LR_group':'LR_gxt'] = (p_group_LR, p_time_LR, p_gxt_LR)


            # Could also have tried modelling the trmt and ctrl groups separately
            #   not using this -- LLR test seems best and most standard
            # modl_data_ctrl = single_roi_data[self.ctrl_bool__data].copy()
            # modl_data_trmt = single_roi_data[~self.ctrl_bool__data].copy()

            # # check responders separately
            # modl_data_resp = single_roi_data[self.resp_bool__data]

            # modl_pymer_resp = pm4dev.Lmer('roidata ~ 1 + visit_week + (1 + visit_week|subjid)', data = modl_data_resp)
            # modl_pymer_resp.fit()

            print_log("", logfile)

        # clean up parallel processing pool
        pool.close()
        pool.join()


        # Add pvals as an attribute
        print("\n" + "Adding attributes: self.lmer_pvals, self.lmer_modls_2g, self.lmer_modls_3g")
        self.lmer_pvals = lmer_pvals
        self.lmer_modls_2g = lmer_modls_2g
        self.lmer_modls_3g = lmer_modls_3g


    def lmer_fdr(self, view=True):
        """Control for multiple comparisons of LLR test with FDR. Also perform post-hoc
        Tukey HSD tests on data, calculate deltas and effect sizes, generate plots and
        output tables for this metric.
        """

        assert (self.lmer_pvals is not None), "Run model_lmer() first."

        # Define and refresh log file and results table
        logfile = self.roidata_dir + "lmer_fdr_" + self.metric + "_log.txt"
        if os.path.isfile(logfile):
            os.remove(logfile)

        restable_file = self.roidata_dir + "lmer_fdr_" + self.metric + "_results_df.pkl"
        if os.path.isfile(restable_file):
            os.remove(restable_file)

        # create directory for post-hoc distrib. plots
        lmer_plot_dir = self.roidata_dir + "LMER_plots/"
        if not os.path.isdir(lmer_plot_dir):
            os.mkdir(lmer_plot_dir)

        # Create dataframe for corrected p-values and H0 rejection bools
        #   start with group, time and gxt will be added later
        #   dtypes will be assigned at time of column value assignment
        lmer_fdr_pvals = pd.DataFrame(index=self.lmer_pvals.index,
                                      columns=["LR_group", "LRg_reject", "LRg_reject_soft"])


        # Correct LR_group pvals for multiple comparisons with FDR
        #   N.B. lmer_pvals has ROIs as index, values are all p-values
        reject, pcorr, _, _ = sm.stats.multipletests(self.lmer_pvals["LR_group"].values,
                                                     alpha=0.05, method='fdr_tsbh')

        # report rejections of H0 (H0 = no effect of adding the group parameter
        #   to the model)
        print_log("Any effects of LR_group at p < 0.05?  {any} ({sum})".format(any=np.any(reject), sum=np.sum(reject)), logfile)

        # check for rejections at alpha=0.10
        reject_more = (pcorr < 0.10) & (pcorr >= 0.05)
        reject_group_soft = pcorr < 0.10
        print_log("Any effects of LR_group at 0.05 < p < 0.10?  {any} ({sum})".format(any=np.any(reject_more), sum=np.sum(reject_more)), logfile)

        lmer_fdr_pvals["LR_group"] = pcorr
        lmer_fdr_pvals["LRg_reject"] = reject
        lmer_fdr_pvals["LRg_reject_soft"] = reject_group_soft

        print_log("", logfile)


        # Check group post-hocs, if there is a (soft) group effect
        #   Note the likelihood ratio test that indicated a group effect for (some)
        #   ROIs is similar to doing an ANOVA or F-test, like
        #   F, p = stats.f_oneway(ctrl_fe_vals, resp_fe_vals, nonr_fe_vals)
        #   to identify a group effect. Should report as "...politeness affected pitch
        #   (χ2(1)=11.62, p=0.00065), lowering it by about 19.7 Hz ± 5.6 (standard errors)..."
        #   Now investigate these ROIs with significant group effects to see which
        #   groups actually differ within the ROIs.

        # Use a dataframe to hold the main results table from this metric
        #   leaving the index as default for now, MultiIndex will be created in the
        #   combined df below
        restable_columns = ["ROI", "Groups", "Metric", "Direction",
                            "Delta", "Cohen_d", "LLR_FDR-p", "Also_HSD", "Grp_FDR-p"]
        results_table = pd.DataFrame(columns=restable_columns)

        # Use a dict to store the tukey-HSD results
        tukeyhsd_results = {}

        # Also create dataframe with single trigroup column to store the individual
        #   subject's predicted values from the model
        ctrls_sorted = self.subj_bools_chrt['ctrl'].sort_index()
        resps_sorted = self.subj_bools_chrt['resp'].sort_index()
        nonrs_sorted = self.subj_bools_chrt['nonr'].sort_index()

        fixef_df = pd.DataFrame(index=ctrls_sorted.index, columns=['trigroup'])

        # generate trigroup variable to categorize each subject
        fixef_df.loc[ctrls_sorted, 'trigroup'] = 'ctrl'
        fixef_df.loc[resps_sorted, 'trigroup'] = 'resp'
        fixef_df.loc[nonrs_sorted, 'trigroup'] = 'nonr'


        def roi_violinplot(ax, crn_vals, crn_ests, crn_ses, rejarray):
            # Generate a nice plot of data points and distrib, with mean and CIs
            # Plot ctrl, resp, nonr data points as [1,2,3]

            # plot all data points without jitter
            plt_kwargs={'ls': 'None', 'marker': 'o', 'ms': 3.5, 'mfc': 'None',
                        'mec': (0.,0.,0.,0.33), 'mew': 1.25,
                        'zorder': 2, 'label': None}
            # x = plt_jitter(np.array([1]), len(crn_vals[0]), jitrng=0.17, yvals=crn_vals[0])
            # ax.plot(x, crn_vals[0], **plt_kwargs)
            # x = plt_jitter(np.array([2]), len(crn_vals[1]), jitrng=0.17, yvals=crn_vals[1])
            # ax.plot(x, crn_vals[1], **plt_kwargs)
            # x = plt_jitter(np.array([3]), len(crn_vals[2]), jitrng=0.17, yvals=crn_vals[2])
            # ax.plot(x, crn_vals[2], **plt_kwargs)
            ax.plot(len(crn_vals[0])*[1.], crn_vals[0], **plt_kwargs)
            ax.plot(len(crn_vals[1])*[2.], crn_vals[1], **plt_kwargs)
            ax.plot(len(crn_vals[2])*[3.], crn_vals[2], **plt_kwargs)


            # Violin plots to show distribution
            #   for BW selection notes, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.set_bandwidth.html
            ax.set_prop_cycle(cycler('color', ['k', 'k', 'k']))     # all black
            kde_ctrl = stats.gaussian_kde(crn_vals[0])
            vparts = ax.violinplot([crn_vals[0], crn_vals[1], crn_vals[2]],
                                   positions=[1, 2, 3], widths=0.67,
                                   showmeans=False, showmedians=False, showextrema=False,
                                   bw_method=0.6*kde_ctrl.factor)
            for pc in vparts['bodies']:
                pc.set_facecolor((0.,0.36,0.57,0.5))    # blue violin bodies
                # pc.set_edgecolor('none')

            # Plot estimates as hlines and SEs as errorbars (not boxes)
            # plt_kwargs={'lw': 1.5, 'colors': '#005c91', 'zorder': 5}
            plt_kwargs={'lw': 1.25, 'colors': '#000000', 'zorder': 5}
            ax.hlines(crn_ests[0], 0.625, 1.375, **plt_kwargs)
            ax.hlines(crn_ests[1], 1.625, 2.375, **plt_kwargs)
            ax.hlines(crn_ests[2], 2.625, 3.375, **plt_kwargs)

            # plt_kwargs={'lw': 1.6, 'edgecolor': '#9b5d11',
            #             'fill': False,
            #             'zorder': 4}
            # ctrl_serect = plt.matplotlib.patches.Rectangle(xy=(0.8, crn_ests[0] - ctrl_se),
            #                                                width=0.4, height=2*ctrl_se, **plt_kwargs)
            # ax.add_patch(ctrl_serect)
            # resp_serect = plt.matplotlib.patches.Rectangle(xy=(1.8, crn_ests[1] - resp_se),
            #                                                width=0.4, height=2*resp_se, **plt_kwargs)
            # ax.add_patch(resp_serect)
            # nonr_serect = plt.matplotlib.patches.Rectangle(xy=(2.8, crn_ests[2] - nonr_se),
            #                                                width=0.4, height=2*nonr_se, **plt_kwargs)
            # ax.add_patch(nonr_serect)

            # brown: '#7c4706'
            # blue: '#005c91'
            plt_kwargs={'elinewidth': 1.0, 'ecolor': '#000000', 'alpha': 0.67,
                        'capsize': 8, 'capthick': 1.25,
                        'zorder': 4}
            ax.errorbar(1, crn_ests[0], yerr=crn_ses[0], fmt='none', **plt_kwargs)
            ax.errorbar(2, crn_ests[1], yerr=crn_ses[1], fmt='none', **plt_kwargs)
            ax.errorbar(3, crn_ests[2], yerr=crn_ses[2], fmt='none', **plt_kwargs)


            # # Draw translucent rectangles for the CIs
            # #   Note the intervals are symmetric, but I'm keeping the calculations
            # ctrl_ci025 = modl_3g.coefs.loc['(Intercept)', '2.5_ci']
            # ctrl_ci975 = modl_3g.coefs.loc['(Intercept)', '97.5_ci']
            # ctrl_ciheight = [ctrl_ci975 - ctrl_ci025]

            # resp_ci025 = modl_3g.coefs.loc['trigroupresp', '2.5_ci']
            # resp_ci975 = modl_3g.coefs.loc['trigroupresp', '97.5_ci']
            # resp_ciheight = [resp_ci975 - resp_ci025]
            # resp_cilow = crn_ests[0] + resp_ci025

            # nonr_ci025 = modl_3g.coefs.loc['trigroupnonr', '2.5_ci']
            # nonr_ci975 = modl_3g.coefs.loc['trigroupnonr', '97.5_ci']
            # nonr_ciheight = [nonr_ci975 - nonr_ci025]
            # nonr_cilow = crn_ests[0] + nonr_ci025

            # plt_kwargs={'facecolor': (0.5, 0.5, 0.5, 0.2),
            #             'edgecolor': 'None',
            #             'fill': True,
            #             'zorder': 1}
            # ctrl_cirect = plt.matplotlib.patches.Rectangle(xy=(0.67, ctrl_ci025),
            #                                                width=0.66, height=ctrl_ciheight, **plt_kwargs)
            # ax.add_patch(ctrl_cirect)
            # resp_cirect = plt.matplotlib.patches.Rectangle(xy=(1.67, resp_cilow),
            #                                                width=0.66, height=resp_ciheight, **plt_kwargs)
            # ax.add_patch(resp_cirect)
            # nonr_cirect = plt.matplotlib.patches.Rectangle(xy=(2.67, nonr_cilow),
            #                                                width=0.66, height=nonr_ciheight, **plt_kwargs)
            # ax.add_patch(nonr_cirect)


            # Ticks and labels -- indicate significance here
            labels = ["CTRL", "RESP", "NONR"]

            # Now using HSD code below to indicate significance
            # labels[0] += "\n\n"     # for consistency of axis sizes across plots
            # if pcorr[0] < 0.1:
            #     labels[1] += r"$^\bigstar$" + "\n" + r"$p_{ctrl}$" + "={p:0.3f}".format(p=pcorr[0])

            # if (pcorr[1] < 0.1) and (pcorr[2] < 0.1):
            #     labels[2] += r"$^{\bigstar\dag}$" + "\n" + r"$p_{ctrl}$" + "={p:0.3f}\n".format(p=pcorr[1]) + r"$p_{resp}$" + "={p:0.3f}".format(p=pcorr[2])
            # elif pcorr[1] < 0.1:
            #     labels[2] += r"$^\bigstar$" + "\n" + r"$p_{ctrl}$" + "={p:0.3f}".format(p=pcorr[1])
            # elif pcorr[2] < 0.1:
            #     labels[2] += r"$^\dag$" + "\n" + r"$p_{resp}$" + "={p:0.3f}".format(p=pcorr[2])

            # Use HSD results to indicate significance
            # draw horizontal bar to indicate significance
            # see https://matplotlib.org/tutorials/text/annotations.html#plotting-guide-annotation
            if np.any(rejarray):
                shortbar_props = {'arrowstyle': '-', 'connectionstyle': 'bar,fraction=-0.1', 'shrinkA': 0., 'shrinkB': 0.}
                longbar_props = {'arrowstyle': '-', 'connectionstyle': 'bar,fraction=-0.047', 'shrinkA': 0., 'shrinkB': 0.}
                bbox_props = {'boxstyle': 'square', 'fc': "white", 'ec': 'none', 'mutation_aspect': 0.5}
                for gd, p in zip(roi_results['Groups'].loc[roi_results['Also_HSD']],
                                 roi_results['Grp_FDR-p'].loc[roi_results['Also_HSD']]):
                    if p < 0.001:
                        s = r"$\bigstar\bigstar\bigstar$"
                    elif p < 0.01:
                        s = r"$\bigstar\bigstar$"
                    else:
                        s = r"$\bigstar$"

                    ygap = -0.11    # level to plot tips of signif. bars
                    if gd == 'R -- C':
                        # Plot the bar, then cover with a white box and the star(s)
                        ax.annotate('', xy=(0.167, ygap), xytext=(0.48, ygap), xycoords='axes fraction', arrowprops=shortbar_props)
                        ax.annotate(s, xy=(0.3235, ygap - 0.03), xycoords='axes fraction', ha='center', va='center', bbox=bbox_props)

                    elif gd == 'N -- C':
                        # leave space above only if other signif results present
                        if len(roi_results['Groups'].loc[roi_results['Also_HSD']]) > 1:
                            ygap -= 0.06

                        ax.annotate('', xy=(0.167, ygap), xytext=(0.833, ygap), xycoords='axes fraction', arrowprops=longbar_props)
                        ax.annotate(s, xy=(0.5, ygap - 0.03), xycoords='axes fraction', ha='center', va='center', bbox=bbox_props)

                    elif gd == 'N -- R':
                        ax.annotate('', xy=(0.52, ygap), xytext=(0.833, ygap), xycoords='axes fraction', arrowprops=shortbar_props)
                        ax.annotate(s, xy=(0.6765, ygap - 0.03), xycoords='axes fraction', ha='center', va='center', bbox=bbox_props)

                    else:
                        raise ValueError("gd not found: {}".format(gd))

            ax.set_xticks([1,2,3])
            ax.set_xticklabels(labels)
            ax.set_xlim(0.5, 3.5)

            # horizontal grid lines
            ax.yaxis.grid(True, lw=0.75, ls=':', color='black', alpha=0.20)


        # Iterate through rois with a (soft) group effect
        roiLR = None    # track prev ROI to make figures in pairs
        rois_to_posthoc = lmer_fdr_pvals.index[reject_group_soft].tolist()
        rois_to_plot = list(rois_to_posthoc)    # make a copy so we can drop items
        for roi in rois_to_posthoc:
            print_log("Inspecting ROI: {roi}".format(roi=roi), logfile)

            modl_3g = self.lmer_modls_3g[roi]
            # modl_3g.summary()
            # Formula: roidata ~ 1 + trigroup + visit_week + trigroup:visit_week + (1 + visit_week|subjid)
            # Number of observations: 702  Groups: {'subjid': 234.0}
            # Log-likelihood: 2143.002     AIC: -4286.004
            #
            # Random effects:
            #                  Name  Var    Std
            # subjid    (Intercept)  0.0  0.017
            # subjid     visit_week  0.0  0.000
            # Residual               0.0  0.006
            #
            #                 IV1         IV2   Corr
            # subjid  (Intercept)  visit_week  0.013
            #
            # Fixed effects:
            #                          Estimate  2.5_ci  97.5_ci     SE       DF   T-stat  P-val  Sig
            # (Intercept)                 0.499   0.496    0.503  0.002  231.001  269.002  0.000  ***
            # trigroupresp                0.005  -0.000    0.011  0.003  231.000    1.886  0.061    .
            # trigroupnonr               -0.003  -0.009    0.002  0.003  231.000   -1.128  0.261
            # visit_week                 -0.000  -0.000    0.000  0.000  231.000   -0.204  0.839
            # trigroupresp:visit_week     0.000  -0.000    0.000  0.000  231.000    0.532  0.595
            # trigroupnonr:visit_week     0.000  -0.000    0.000  0.000  231.000    0.001  0.999

            # Note the design matrix is of the style where the intercept is specified to
            # represent the control group, while resp and nonr get dummy coded:
            # modl_3g.design_matrix.head()
            #    (Intercept)  trigroupresp  trigroupnonr  visit_week  trigroupresp:visit_week  trigroupnonr:visit_week
            # 0          1.0           0.0           0.0         0.0                      0.0                      0.0
            # 1          1.0           0.0           0.0         2.0                      0.0                      0.0
            # 2          1.0           0.0           0.0         8.0                      0.0                      0.0
            # 3          1.0           1.0           0.0         0.0                      0.0                      0.0
            # 4          1.0           1.0           0.0         2.0                      2.0                      0.0


            # First, get the p-values directly from the model fit/Chi-squared test
            # Note these are identical to calculating the values using eqn 4.14 of
            # Galwey2014. The SE values reported in the summary() above, which are used
            # to calculate these p-values should be the SE of the *difference* between
            # means for these two terms, but the (Intercept) (ctrl) SE would be the SE
            # of the mean itself, I think.
            resp_vs_ctrl_pval = self.lmer_pvals.loc[roi, '3G_resp']
            nonr_vs_ctrl_pval = self.lmer_pvals.loc[roi, '3G_nonr']

            # Need to calculate t-value to test the null hypothesis of no differences
            # btw resp and nonr.
            # Could also use the CI's, but a p-value is probably easier to publish
            ctrl_est = modl_3g.coefs.loc['(Intercept)', 'Estimate']
            resp_est = ctrl_est + modl_3g.coefs.loc['trigroupresp', 'Estimate']
            nonr_est = ctrl_est + modl_3g.coefs.loc['trigroupnonr', 'Estimate']
            delta = nonr_est - resp_est
            se_avg = 1/2.*(modl_3g.coefs.loc['trigroupnonr', 'SE'] + modl_3g.coefs.loc['trigroupresp', 'SE'])
            df = modl_3g.coefs.loc['(Intercept)', 'DF']

            # H0: delta = 0
            # t = (delta_hat - delta)/SE_deltahat
            t_stat = delta/se_avg
            nonr_vs_resp_pval = 2.*stats.t.sf(np.abs(t_stat), df)  # two-tailed

            # Do FDR correction for the 3 p-values
            # using fdr_bh for so few values (aka just 'fdr' in R's p-adjust)
            reject, pcorr, _, _ = sm.stats.multipletests([resp_vs_ctrl_pval, nonr_vs_ctrl_pval, nonr_vs_resp_pval],
                                                          alpha=0.10, method='fdr_bh')

            # Report pairwise comparison results
            print_log("Pair-wise comparisons of 3G group effects", logfile)
            print_log("   contrast     delta   p-value    fdr-p   rej-H0", logfile)
            print_log("resp - ctrl   {delta:+01.4f}    {pval:01.4f}   {fdrp:01.4f}   {rej}".format(delta=resp_est-ctrl_est, pval=resp_vs_ctrl_pval, fdrp=pcorr[0], rej=reject[0]), logfile)
            print_log("nonr - ctrl   {delta:+01.4f}    {pval:01.4f}   {fdrp:01.4f}   {rej}".format(delta=nonr_est-ctrl_est, pval=nonr_vs_ctrl_pval, fdrp=pcorr[1], rej=reject[1]), logfile)
            print_log("nonr - resp   {delta:+01.4f}    {pval:01.4f}   {fdrp:01.4f}   {rej}".format(delta=nonr_est-resp_est, pval=nonr_vs_resp_pval, fdrp=pcorr[2], rej=reject[2]), logfile)
            print_log("", logfile)


            # Extract fixed effects data for each subject to test/plot
            # In this way, each subject in nonr group gets the same value for the
            #   group slope and intercept, but every subject has a unique overall
            #   (Intercept) and visit_week value; so each subject gets unique estimated
            #   values overall, after doing the algebra.
            # Note the 'estimates', e.g. ctrl_est above, are the means of the
            #   group data, so e.g. ctrl_est = ctrl_fe_vals.mean()

            ctrl_fe_vals = modl_3g.fixef.loc[ctrls_sorted, '(Intercept)']     # note fixef organized by subject, alpha-numerically sorted
            resp_fe_vals = np.sum(modl_3g.fixef.loc[resps_sorted, ['(Intercept)', 'trigroupresp']], axis=1)
            nonr_fe_vals = np.sum(modl_3g.fixef.loc[nonrs_sorted, ['(Intercept)', 'trigroupnonr']], axis=1)

            ctrl_se = modl_3g.coefs.loc['(Intercept)', 'SE']
            resp_se = modl_3g.coefs.loc['trigroupresp', 'SE']
            nonr_se = modl_3g.coefs.loc['trigroupnonr', 'SE']

            # create series variable with fe values and add it to the df
            fe_vals = pd.Series(index=ctrls_sorted.index, name=roi, dtype=float)

            fe_vals.loc[ctrl_fe_vals.index] = ctrl_fe_vals
            fe_vals.loc[resp_fe_vals.index] = resp_fe_vals
            fe_vals.loc[nonr_fe_vals.index] = nonr_fe_vals

            fixef_df[roi] = fe_vals


            # Show Tukey HSD results for group tests
            # N.B. this seems to be a more common method than FDR on 3 p-values,
            #      so this is preferred, I guess
            # tukeyhsd_result = sm.stats.multicomp.pairwise_tukeyhsd(fixef_df[roi], fixef_df['trigroup'])

            # This runs my adapted code so p-values are returned in tukeyhsd_result.p_adjs
            tukeyhsd_result = mcdev.pairwise_tukeyhsd(fixef_df[roi], fixef_df['trigroup'])
            print_log("Compare Tukey HSD test:", logfile)
            print_log(tukeyhsd_result.summary().as_text(), logfile)
            print_log("", logfile)
            tukeyhsd_results[roi] = tukeyhsd_result


            # If true differences btw groups exist, add them to the results table
            # N.B. this will add effects with btw-group FDR-p < 0.10 to the table,
            #      but it is later filtered to include only results with
            #      FDR-p < 0.05 and HSD = True
            if np.any(reject):
                # "Groups", "LLR_FDR-p", "Also_HSD", "Grp_FDR-p", "Delta", "Direction"
                grp_diff_col = []
                also_HSD_col = []
                delta_col = []
                cohd_col = []
                FDRp_col = []
                direction_col = []

                if reject[0]:
                    grp_diff_col.append("R -- C")

                    if tukeyhsd_result.reject[1]:   # order of tukey is in ctrl-nonr, ctrl-resp, nonr-resp
                        also_HSD_col.append(True)
                    else:
                        also_HSD_col.append(False)

                    FDRp_col.append(pcorr[0])

                    delta = resp_est - ctrl_est
                    delta_col.append(delta)

                    if delta >= 0:
                        direction_col.append(u"R ▲")
                    else:
                        direction_col.append(u"R ▼")

                    cohd_col.append(cohen_d(resp_fe_vals, ctrl_fe_vals))

                if reject[1]:
                    grp_diff_col.append("N -- C")

                    if tukeyhsd_result.reject[0]:
                        also_HSD_col.append(True)
                    else:
                        also_HSD_col.append(False)

                    FDRp_col.append(pcorr[1])

                    delta = nonr_est - ctrl_est
                    delta_col.append(delta)
                    if delta >= 0:
                        direction_col.append(u"N ▲")
                    else:
                        direction_col.append(u"N ▼")

                    cohd_col.append(cohen_d(nonr_fe_vals, ctrl_fe_vals))

                if reject[2]:
                    grp_diff_col.append("N -- R")

                    if tukeyhsd_result.reject[2]:
                        also_HSD_col.append(True)
                    else:
                        also_HSD_col.append(False)

                    FDRp_col.append(pcorr[2])

                    delta = nonr_est - resp_est
                    delta_col.append(delta)
                    if (delta >= 0) and (resp_est > ctrl_est):
                        direction_col.append(u"N ▲")
                    elif (delta >= 0) and (nonr_est < ctrl_est):
                        direction_col.append(u"R ▼")
                    elif (delta >= 0):
                        direction_col.append(u"N ▲/R ▼")
                    elif (delta < 0) and (nonr_est > ctrl_est):
                        direction_col.append(u"R ▲")
                    elif (delta < 0) and (resp_est < ctrl_est):
                        direction_col.append(u"N ▼")
                    else:
                        direction_col.append(u"R ▲/N ▼")

                    cohd_col.append(cohen_d(nonr_fe_vals, resp_fe_vals))

                rej_cnt = np.sum(reject)
                roi_col = [roi]*rej_cnt

                LLR_FDRp = lmer_fdr_pvals.loc[roi, "LR_group"]
                LLR_FDRp_col = [LLR_FDRp]*rej_cnt

                metric_col = [self.metric]*rej_cnt

                roi_results = pd.DataFrame(data={"ROI": roi_col,
                                                 "Groups": grp_diff_col,
                                                 "Metric": metric_col,
                                                 "Direction": direction_col,
                                                 "Delta": delta_col,
                                                 "Cohen_d": cohd_col,
                                                 "LLR_FDR-p": LLR_FDRp_col,
                                                 "Also_HSD": also_HSD_col,
                                                 "Grp_FDR-p": FDRp_col},
                                           columns=restable_columns)

                results_table = results_table.append(roi_results, ignore_index=True)


            # Old ideas:
            # Could test the null hypothesis of no group differences with
            #   an F test -- this is redundant
            # from scipy import stats as st
            # F, p = stats.f_oneway(ctrl_fe_vals, resp_fe_vals, nonr_fe_vals)

            # Could do ANOVA using linear model within this ROI
            #   This is also redundant and a little suspicious, given its not
            #   a mixed model

            # ph_lm = sm.formula.ols('fe_vals ~ trigroup', data=fixef_df).fit()
            # #ph_lm.summary()

            # aov_table = sm.stats.anova_lm(ph_lm, typ='II') # Type 2 ANOVA DataFrame
            # print(aov_table)
            #             sum_sq     df         F    PR(>F)
            # trigroup  0.002605    2.0  4.766774  0.009363
            # Residual  0.063118  231.0       NaN       NaN

            # Could also do separate tukey HDR and t-tests with fdr

            # tkres = sm.stats.multicomp.pairwise_tukeyhsd(fixef_df.fe_vals, fixef_df.trigroup)
            # print(tkres.summary())
            # Multiple Comparison of Means - Tukey HSD,FWER=0.05
            # ============================================
            # group1 group2 meandiff  lower  upper  reject
            # --------------------------------------------
            #  ctrl   nonr  -0.0031  -0.0092 0.003  False
            #  ctrl   resp   0.0053  -0.0009 0.0116 False
            #  nonr   resp   0.0085   0.0019 0.015   True
            # --------------------------------------------

            # or, equivalent output from
            # mcres = sm.stats.multicomp.MultiComparison(fixef_df.fe_vals, fixef_df.trigroup)
            # print(mcres.tukeyhsd())

            # fdr_tsbh
            # pairres = mcres.allpairtest(scipy.stats.ttest_ind, method='fdr_tsbh')
            # print(pairres[0])
            # Test Multiple Comparison ttest_ind
            # FWER=0.05 method=fdr_tsbh
            # alphacSidak=0.02, alphacBonf=0.017
            # =============================================
            # group1 group2   stat   pval  pval_corr reject
            # ---------------------------------------------
            #  ctrl   nonr   1.2916 0.1983   0.1322  False
            #  ctrl   resp  -1.9278 0.0557   0.0557  False
            #  nonr   resp  -3.0071 0.0031   0.0062   True
            # ---------------------------------------------


            # Plot values for significant groups
            # modl_3g.plot_summary(intercept=False, xlim=[-0.01, 0.01])
            # N.B. boxplots use median values, so don't represent the actual
            #   pairwise comparisons being made
            # fixef_df.boxplot('fe_vals', by='trigroup', figsize=(6, 4.5))

            # N.B. it might be nice to split out the plotting function and make
            #   LR plots for all metrics in the ROIs that are known to have
            #   significant effects.

            # Combined plot with L and R ROIs or single plot
            plt.rcParams['font.size'] = 9.
            plt.rcParams['axes.titlepad'] = 3.0
            plt.rcParams['axes.labelpad'] = 2.0
            plt.rcParams['ytick.major.pad'] = 1.5
            plt.rcParams['xtick.major.pad'] = 2.0
            # originally had 6 x 3.25 in for the figsize, but need a 95 mm width for biol. psych.
            # note BP wants either PDF or 1000 DPI raster
            figw = 3.74015
            figh = 2.3622

            # Title axes when necessary, otherwise leave it off
            write_title = False
            if (self.metric == 'FA') \
                or (self.metric == 'L1' and (roi.startswith('Cingulum') \
                                          or roi.startswith('Posterior_thalamic_radiation') \
                                          or roi.startswith('Posterior_limb_of_internal_capsule'))):
                write_title = True

            if roi.endswith('_R') and (roi[:-1] + 'L' in rois_to_plot):
                # Start a combined plot for this R ROI
                roiLR = roi
                figLR, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(figw, figh), dpi=150)
                roi_violinplot(ax2, (ctrl_fe_vals, resp_fe_vals, nonr_fe_vals),
                                    (ctrl_est, resp_est, nonr_est),
                                    (ctrl_se, resp_se, nonr_se),
                                    reject)

                if write_title:
                    ax2.set_title("{}".format(abbrev_roi(roi)))
                # bbox_props = {'boxstyle': 'round', 'fc': 'white', 'ec': '0.5'}
                # ax2.annotate(abbrev_roi(roi), xy=(0.50, 1.00), xycoords='axes fraction', ha='center', va='center', bbox=bbox_props)

            elif roi.endswith('_L') and (roiLR is not None) and (roi[:-1] == roiLR[:-1]):
                # Finish the combined plot
                roi_violinplot(ax1, (ctrl_fe_vals, resp_fe_vals, nonr_fe_vals),
                                    (ctrl_est, resp_est, nonr_est),
                                    (ctrl_se, resp_se, nonr_se),
                                    reject)
                ax1.set_ylabel(self.metric_label)

                if write_title:
                    ax1.set_title("{}".format(abbrev_roi(roi)))
                # bbox_props = {'boxstyle': 'round', 'fc': 'white', 'ec': '0.5'}
                # ax1.annotate(abbrev_roi(roi), xy=(0.50, 1.00), xycoords='axes fraction', ha='center', va='center', bbox=bbox_props)

                figLR.tight_layout()

                # figLR.subplots_adjust(left=0.10, bottom=0.16, right=0.98, top=0.92, wspace=0.07)
                figLR.subplots_adjust(left=0.15, bottom=0.175, right=0.98, top=0.935, wspace=0.07)
                figLR.canvas.draw()
                figLR.savefig(lmer_plot_dir + "group_effect-" + roi[:-1] + "LR_" + self.metric + ".pdf", dpi=1000)

                if view:
                    figLR.show()
                    raw_input("Press Enter to continue...")
                else:
                    plt.close(figLR)

            else:
                # Start a dual plot for this ROI
                figLR, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(figw, figh), dpi=150)
                if roi.endswith('_R'):
                    this_ax = ax2
                    other_ax = ax1
                    roi_2 = roi[:-1] + 'L'
                elif roi.endswith('_L'):
                    this_ax = ax1
                    other_ax = ax2
                    roi_2 = roi[:-1] + 'R'
                else:
                    # corpus callosum, fornix
                    this_ax = ax1
                    other_ax = None

                roi_violinplot(this_ax, (ctrl_fe_vals, resp_fe_vals, nonr_fe_vals),
                                        (ctrl_est, resp_est, nonr_est),
                                        (ctrl_se, resp_se, nonr_se),
                                        reject)

                # Plot its partner if it has one, but the partner shouldn't be posthoc tested
                # this code comes from ~ l. 2813 above
                if other_ax is not None:
                    modl_3g_2 = self.lmer_modls_3g[roi_2]

                    ctrl_fe_vals_2 = modl_3g_2.fixef.loc[ctrls_sorted, '(Intercept)']     # note fixef organized by subject, alpha-numerically sorted
                    resp_fe_vals_2 = np.sum(modl_3g_2.fixef.loc[resps_sorted, ['(Intercept)', 'trigroupresp']], axis=1)
                    nonr_fe_vals_2 = np.sum(modl_3g_2.fixef.loc[nonrs_sorted, ['(Intercept)', 'trigroupnonr']], axis=1)

                    ctrl_est_2 = modl_3g_2.coefs.loc['(Intercept)', 'Estimate']
                    resp_est_2 = ctrl_est_2 + modl_3g_2.coefs.loc['trigroupresp', 'Estimate']
                    nonr_est_2 = ctrl_est_2 + modl_3g_2.coefs.loc['trigroupnonr', 'Estimate']

                    ctrl_se_2 = modl_3g_2.coefs.loc['(Intercept)', 'SE']
                    resp_se_2 = modl_3g_2.coefs.loc['trigroupresp', 'SE']
                    nonr_se_2 = modl_3g_2.coefs.loc['trigroupnonr', 'SE']

                    reject_2 = np.array([False, False, False])

                    roi_violinplot(other_ax, (ctrl_fe_vals_2, resp_fe_vals_2, nonr_fe_vals_2),
                                             (ctrl_est_2, resp_est_2, nonr_est_2),
                                             (ctrl_se_2, resp_se_2, nonr_se_2),
                                             reject_2)

                ax1.set_ylabel(self.metric_label)

                if write_title:
                    this_ax.set_title("{}".format(abbrev_roi(roi)))

                    if other_ax is not None:
                        other_ax.set_title("{}".format(abbrev_roi(roi_2)))

                figLR.tight_layout()

                # figLR.subplots_adjust(left=0.10, bottom=0.16, right=0.98, top=0.92, wspace=0.07)
                figLR.subplots_adjust(left=0.15, bottom=0.175, right=0.98, top=0.935, wspace=0.07)
                figLR.canvas.draw()
                figLR.savefig(lmer_plot_dir + "group_effect-" + roi[:-1] + "LR_" + self.metric + ".pdf", dpi=1000)

                if view:
                    figLR.show()
                    raw_input("Press Enter to continue...")
                else:
                    plt.close(figLR)

        print("\n")


        # Check time p-vals for ROIs where there is a (soft) group effect
        # We are interested in the "LR_time", "LR_gxt" lmer_pvals columns
        #   to correct for multiple comparisons and check for significance.
        # Then probably interested in the "3G_time", "3G_nxt", "3G_rxt"
        #   columns.

        # H0: no significant slope associated with time across all subjects
        check_pvals = self.lmer_pvals.loc[reject_group_soft, "LR_time"].values

        reject, pcorr, _, _ = sm.stats.multipletests(check_pvals,
                                                     alpha=0.05, method='fdr_tsbh')

        print_log("Any effects of LR_time within group at p < 0.05?  {any} ({sum})".format(any=np.any(reject), sum=np.sum(reject)), logfile)

        # also check for rejections at alpha=0.10
        reject_more = (pcorr < 0.10) & (pcorr >= 0.05)
        reject_soft = pcorr < 0.10
        print_log("Any effects of LR_time at 0.05 < p < 0.10?  {any} ({sum})".format(any=np.any(reject_more), sum=np.sum(reject_more)), logfile)

        lmer_fdr_pvals.loc[reject_group_soft, "LR_time"] = pcorr
        lmer_fdr_pvals.loc[reject_group_soft, "LRt_reject"] = reject
        lmer_fdr_pvals.loc[reject_group_soft, "LRt_reject_soft"] = reject_soft

        print_log("", logfile)


        # H0: no significant effect of estimating a slope in any group
        check_pvals = self.lmer_pvals.loc[reject_group_soft, "LR_gxt"]

        reject, pcorr, _, _ = sm.stats.multipletests(check_pvals,
                                                     alpha=0.05, method='fdr_tsbh')

        print_log("Any effects of LR_gxt within group at p < 0.05?  {any} ({sum})".format(any=np.any(reject), sum=np.sum(reject)), logfile)

        # also check for rejections at alpha=0.10
        reject_more = (pcorr < 0.10) & (pcorr >= 0.05)
        reject_soft = pcorr < 0.10
        print_log("Any effects of LR_gxt at 0.05 < p < 0.10?  {any} ({sum})".format(any=np.any(reject_more), sum=np.sum(reject_more)), logfile)

        lmer_fdr_pvals.loc[reject_group_soft, "LR_gxt"] = pcorr
        lmer_fdr_pvals.loc[reject_group_soft, "LRgxt_reject"] = reject
        lmer_fdr_pvals.loc[reject_group_soft, "LRgxt_reject_soft"] = reject_soft

        print_log("", logfile)


        # Really we're interested in slopes in the responders group
        # H0: no significant effect of temporal slope in the responders group
        check_pvals = self.lmer_pvals.loc[reject_group_soft, "3G_rxt"]

        reject, pcorr, _, _ = sm.stats.multipletests(check_pvals,
                                                     alpha=0.05, method='fdr_tsbh')

        print_log("Any effects of 3G_rxt within group at p < 0.05?  {any} ({sum})".format(any=np.any(reject), sum=np.sum(reject)), logfile)

        # also check for rejections at alpha=0.10
        reject_more = (pcorr < 0.10) & (pcorr >= 0.05)
        reject_soft = pcorr < 0.10
        print_log("Any effects of 3G_rxt at 0.05 < p < 0.10?  {any} ({sum})".format(any=np.any(reject_more), sum=np.sum(reject_more)), logfile)

        lmer_fdr_pvals.loc[reject_group_soft, "3G_rxt"] = pcorr
        lmer_fdr_pvals.loc[reject_group_soft, "3Grxt_reject"] = reject
        lmer_fdr_pvals.loc[reject_group_soft, "3Grxt_reject_soft"] = reject_soft

        print_log("", logfile)


        # Write out results table with correct dtypes
        results_table['Also_HSD'] = results_table.Also_HSD.astype(bool)
        results_table.to_pickle(restable_file)

        # Add fdr-pvals and tukey_hsd results as attributes
        print("Adding attributes: self.lmer_fdr_pvals, self.lmer_fdr_tukeyhsd_results, self.fixef_df")
        self.lmer_fdr_pvals = lmer_fdr_pvals
        self.lmer_fdr_tukeyhsd_results = tukeyhsd_results

        # and add fixef_df, after reordering columns
        cols = ['trigroup'] + sorted([col for col in fixef_df if col != 'trigroup'])
        self.fixef_df = fixef_df[cols]

