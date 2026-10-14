import React, {useEffect, useRef, useState} from 'react';
import {Activity, AlertTriangle, ArrowUpRight, CircleDot, Info, ShieldAlert, X} from 'lucide-react';
import './alert-details.css';

const fmt = (value, digits=2) => value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(digits);

export default function AlertReview({alerts, alertTotal, setAlertTotal, chartMetrics, experiment, model, setAlerts, setError, setModel, api}) {
 const [selected, setSelected] = useState(null);
 const [detail, setDetail] = useState(null);
 const [detailLoading, setDetailLoading] = useState(false);
 const [detailError, setDetailError] = useState('');
 const [loadingMore, setLoadingMore] = useState(false);
 const requestId = useRef(0);

 async function selectCandidate(candidate) {
  const currentRequest = ++requestId.current;
  setSelected(candidate);
  setDetail(null);
  setDetailError('');
  setDetailLoading(true);
  try {
   const result = await api(`/api/alerts/${encodeURIComponent(candidate.fragment_id)}?model=${encodeURIComponent(candidate.model)}`);
   if (requestId.current === currentRequest) setDetail(result);
  } catch (error) {
   if (requestId.current === currentRequest) setDetailError(error.message);
  } finally {
   if (requestId.current === currentRequest) setDetailLoading(false);
  }
 }

 function closeDetail() {
  requestId.current += 1;
  setSelected(null);
  setDetail(null);
  setDetailError('');
  setDetailLoading(false);
 }

 useEffect(() => {
  if (!selected) return undefined;
  function closeOnEscape(event) {
   if (event.key === 'Escape') closeDetail();
  }
  window.addEventListener('keydown', closeOnEscape);
  return () => window.removeEventListener('keydown', closeOnEscape);
 }, [selected]);

 async function changeModel(event) {
  const nextModel = event.target.value;
  setModel(nextModel);
  closeDetail();
  setAlerts([]);
  setAlertTotal(0);
  try {
   const result = await api('/api/alerts?limit=80&offset=0&model='+encodeURIComponent(nextModel));
   setAlerts(result.alerts || []);
   setAlertTotal(result.total_candidates || 0);
  } catch (error) {
   setError(error.message);
  }
 }

 async function loadMore() {
  setLoadingMore(true);
  try {
   const result = await api(`/api/alerts?limit=80&offset=${alerts.length}&model=${encodeURIComponent(model)}`);
   setAlerts(current => [...current, ...(result.alerts || [])]);
   setAlertTotal(result.total_candidates || 0);
  } catch (error) {
   setError(error.message);
  } finally {
   setLoadingMore(false);
  }
 }

 return <>
  <div className="alert-summary"><div><span className="alert-summary-icon"><ShieldAlert size={21}/></span><div><strong>{experiment?alertTotal.toLocaleString():'—'} anomaly candidates</strong><p>{experiment?`Showing ${alerts.length.toLocaleString()} of ${alertTotal.toLocaleString()} · ranked by score for the selected detector.`:'Run an experiment to generate scored candidates.'}</p></div></div>{experiment&&<select className="select-dark" value={model} onChange={changeModel}>{chartMetrics.map(item=><option key={item.model}>{item.model}</option>)}</select>}</div>
  <div className="panel alerts-panel"><div className="panel-title"><div><h3>Fragment review queue</h3><p>Model-flagged test fragments · human review required</p></div><span className="panel-tag">{experiment?'HELD-OUT TEST':'NO RUN'}</span></div>{experiment&&alerts.length>0?<><div className="alerts-list">{alerts.map((candidate,index)=><div className="alert-row" key={candidate.fragment_id}><div className="alert-index">{String(index+1).padStart(2,'0')}</div><div className="alert-signal"><div className="signal-icon"><Activity size={18}/></div><div><button className="alert-candidate-link" type="button" onClick={()=>selectCandidate(candidate)} aria-label={`View details for ${candidate.fragment_id}`}>{candidate.fragment_id}<ArrowUpRight size={12}/></button><span>Test fragment index {candidate.index_in_test} · {candidate.model}</span></div></div><div className="alert-score"><small>Anomaly score</small><strong>{fmt(candidate.score,4)}</strong><small className="alert-threshold">Threshold {fmt(candidate.threshold,4)}</small></div><span className={`severity ${candidate.severity==='High'?'high':'review'}`}>{candidate.severity==='High'?<AlertTriangle size={13}/>:<CircleDot size={13}/>} {candidate.severity}</span><div className="alert-note">{candidate.note}</div></div>)}</div>{alerts.length<alertTotal&&<div className="alert-load-more"><button className="button secondary" type="button" onClick={loadMore} disabled={loadingMore}>{loadingMore?'Loading…':'Load more candidates'}</button></div>}</>:<div className="mini-empty"><div><Activity size={19}/></div><p>{experiment?'No fragments were flagged by this model at the selected threshold.':'Upload the benchmark CSV and run the model pipeline to populate the review queue.'}</p></div>}<div className="alert-disclaimer"><Info size={15}/><span>Candidate rankings are based on model scores. Severity is a score-ranking label, not the anomaly decision. These flags are not causal explanations or verified hardware failures; check raw telemetry and mission context.</span></div></div>
  {selected&&<div className="candidate-modal-backdrop" onMouseDown={event=>{if(event.target===event.currentTarget)closeDetail()}}><section className="candidate-modal" role="dialog" aria-modal="true" aria-labelledby="candidate-detail-title"><div className="candidate-modal-heading"><div><span className="candidate-modal-eyebrow">HELD-OUT TEST CANDIDATE</span><h2 id="candidate-detail-title">{selected.fragment_id}</h2></div><button className="candidate-close-button" type="button" onClick={closeDetail} aria-label="Close candidate details"><X size={17}/></button></div>{detailLoading?<p className="candidate-detail-message">Loading candidate details…</p>:detailError?<div className="candidate-detail-error"><AlertTriangle size={16}/><span>Candidate details are unavailable: {detailError}</span></div>:detail&&<><div className="candidate-detail-grid"><div><span>Test fragment index</span><strong>{detail.index_in_test ?? '—'}</strong></div><div><span>Selected detector</span><strong>{detail.model || '—'}</strong></div><div><span>Anomaly score</span><strong>{fmt(detail.score,4)}</strong></div><div><span>Applicable threshold</span><strong>{fmt(detail.threshold,4)}</strong></div><div><span>Prediction</span><strong>{detail.prediction || '—'}</strong></div><div><span>Severity</span><strong className={`candidate-severity ${detail.severity==='High'?'high':'review'}`}>{detail.severity || '—'}</strong></div></div><div className="candidate-feature-section"><h3>Telemetry feature values</h3>{detail.features&&Object.keys(detail.features).length>0?<div className="candidate-feature-table-wrap"><table className="candidate-feature-table"><thead><tr><th>Feature</th><th>Stored value</th></tr></thead><tbody>{Object.entries(detail.features).map(([name,value])=><tr key={name}><td>{name}</td><td>{value == null ? '—' : String(value)}</td></tr>)}</tbody></table></div>:<p className="candidate-detail-message">Telemetry feature values are not available for this candidate.</p>}</div></>}<div className="candidate-modal-footer"><button className="button secondary" type="button" onClick={closeDetail}>Close</button></div></section></div>}
 </>;
}
