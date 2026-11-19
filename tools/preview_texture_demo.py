"""Plot verified crystallographic diffraction bundles without rescaling their scientific arrays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from xray_generator.experiment.bundles import inspect_bundle
from xray_generator.experiment.provenance import filesystem_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=filesystem_path(args.dataset)
    rows=[]
    for manifest_path in sorted(data.glob('*/manifest.json')):
        folder=manifest_path.parent
        manifest=inspect_bundle(folder)
        if manifest['profile']!='static_kinematic_gaussian_v1':
            raise ValueError('Expected the static crystallographic diffraction profile.')
        if manifest['readout_id']!='701': continue
        record_path=data/'.runs'/(manifest['run_record_id']+'.json')
        config=json.loads(record_path.read_text(encoding='utf-8'))['configuration']
        detector=config['crystallography']['detector']
        with np.load(folder/'observable.npz',allow_pickle=False) as observable, np.load(folder/'private.npz',allow_pickle=False) as private:
            rows.append((manifest['exposure_id'],observable['counts'].copy(),private['expected_mean'].copy(),detector))
    rows.sort(key=lambda row:row[0])
    if len(rows)!=2: raise ValueError('Expected one selected replicate of each of the two exposures.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    fig.suptitle('Crystallographic diffraction across a Cu / Ti interface\n512 × 512 native detector · 0.1 s exposures',fontsize=17)
    cmap=plt.get_cmap('magma').copy(); cmap.set_bad('#090713')
    vmax=max(float(row[1].max()) for row in rows)
    norm=LogNorm(vmin=.02,vmax=max(vmax,1.))
    summaries=[]
    for index,(identity,counts,mean,detector) in enumerate(rows):
        h,w=mean.shape; pitch=detector['pixel_pitch_mm']; r0,c0=detector['beam_center_row_col']
        extent=[(-.5-c0)*pitch,(w-.5-c0)*pitch,(-.5-r0)*pitch,(h-.5-r0)*pitch]
        for column,(array,title) in enumerate([(mean,'Clean expectation'),(counts,'Native Poisson observation')]):
            ax=axes[index,column]
            plotted=np.ma.masked_less_equal(array,0.)
            im=ax.imshow(plotted,origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='nearest')
            amount=float(array.sum())
            text=f'{amount:,.1f} expected photons' if column==0 else f'{int(amount):,} photons observed'
            ax.set_title(f'{title}\n{text}',fontsize=12)
            ax.set_xlabel('Detector y (mm)')
            ax.set_ylabel(('Boundary crossing' if index==0 else 'Overlapping exposure')+'\nDetector z (mm)')
        summaries.append(dict(exposure_id=identity,shape=list(mean.shape),expected_events=float(mean.sum()),
                              observed_events=int(counts.sum()),readout_id='701'))
    color=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,pad=.02)
    color.set_label('Photons per native pixel · logarithmic display')
    fig.supxlabel('Fixed representative orientations · idealized kinematic scattering · absolute scale uncalibrated',fontsize=10)
    output=filesystem_path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor='white'); plt.close(fig)
    report=output.with_suffix('.json')
    report.write_text(json.dumps(dict(dataset=str(args.dataset.resolve()),profile='static_kinematic_gaussian_v1',
        display='Common logarithmic colour scale; no per-image normalization; scientific arrays unchanged',
        exposures=summaries),indent=2),encoding='utf-8')
    print(json.dumps(dict(preview=str(args.output.resolve()),exposures=summaries),indent=2))


if __name__=='__main__': main()
