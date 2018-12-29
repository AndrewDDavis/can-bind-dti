reload(dmroi)

obj = roidata_objs_excl_outl['FA123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_outl['MD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_outl['L1123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_outl['RD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

dmroi.lmer_results_summary('/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl/')

with gzip.open("group_results/roidata_objs_excl_outl.pkl.gz", mode='wb', compresslevel=6) as pkl_file:
    pickle.dump(roidata_objs_excl_outl, pkl_file)


obj = roidata_objs_excl_sse_outl['FA123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_sse_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_sse_outl['MD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_sse_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_sse_outl['L1123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_sse_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_sse_outl['RD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_sse_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

dmroi.lmer_results_summary('/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_sse_outl/')

with gzip.open("group_results/roidata_objs_excl_sse_outl.pkl.gz", mode='wb', compresslevel=6) as pkl_file:
    pickle.dump(roidata_objs_excl_sse_outl, pkl_file)


obj = roidata_objs_excl_outl_tothemax['FA123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl_tothemax/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_outl_tothemax['MD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl_tothemax/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_outl_tothemax['L1123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl_tothemax/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_excl_outl_tothemax['RD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl_tothemax/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

dmroi.lmer_results_summary('/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-excl_outl_tothemax/')

with gzip.open("group_results/roidata_objs_excl_outl_tothemax.pkl.gz", mode='wb', compresslevel=6) as pkl_file:
    pickle.dump(roidata_objs_excl_outl_tothemax, pkl_file)


obj = roidata_objs_incl_outl['FA123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-incl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_incl_outl['MD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-incl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_incl_outl['L1123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-incl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

obj = roidata_objs_incl_outl['RD123']
obj.roidata_dir = '/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-incl_outl/'
obj.lmer_fdr = dmroi.RoiData.lmer_fdr.__get__(obj)
obj.lmer_fdr(view=False)

dmroi.lmer_results_summary('/Users/andrew/Documents/Research_Projects/CAN-BIND_DTI/group_results/merged_skel_v01v02v03-incl_outl/')

with gzip.open("group_results/roidata_objs_incl_outl.pkl.gz", mode='wb', compresslevel=6) as pkl_file:
    pickle.dump(roidata_objs_incl_outl, pkl_file)

