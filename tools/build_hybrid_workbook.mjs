// Optional XLSX authoring requires the Codex bundled @oai/artifact-tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const directory=path.join(root,'benchmarks/2026-11-17-time-place');
const data=JSON.parse(await fs.readFile(path.join(directory,'data/benchmark.json'),'utf8'));
const provenance=JSON.parse(await fs.readFile(path.join(directory,'provenance.json'),'utf8'));
const wb=Workbook.create();
const overview=wb.worksheets.add('汇总');const cases=wb.worksheets.add('逐条结果');const failures=wb.worksheets.add('未命中');const pairs=wb.worksheets.add('正关联');
const category={time:'时间',place:'地点',both:'时间与地点'};
function table(sheet,rows,widths,height=38){
 sheet.showGridLines=false;const all=sheet.getRangeByIndexes(0,0,rows.length,rows[0].length);all.values=rows;
 all.format={font:{name:'Arial',size:10},rowHeight:height,verticalAlignment:'center'};
 sheet.getRangeByIndexes(0,0,1,rows[0].length).format={fill:'#243B57',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:46,wrapText:true};
 widths.forEach((width,i)=>sheet.getRangeByIndexes(0,i,rows.length,1).format.columnWidth=width);sheet.freezePanes.freezeRows(1);
}
overview.showGridLines=false;overview.mergeCells('A2:L2');overview.getRange('A2').values=[[`时间与地点查询检索对比  ${data.scope.cases} 条用例`]];overview.getRange('A2').format.font={name:'Arial',size:15,bold:true};
const summary=[['范围','方法','用例数','有标签数','命中数','用例命中率','已知fid命中','已知fid总数','微平均覆盖率','宏平均覆盖率','返回关联','待核实']];
for(const [name,group] of [['全部',data.overall],...Object.entries(data.by_category).map(([key,value])=>[category[key],value])]){
 for(const [method,label] of [['baseline','V1视觉＋Qwen'],['hybrid','V2混合＋Qwen']]){
  const m=group[method];summary.push([name,label,m.cases,m.labeled_cases,m.hit_cases,null,m.known_fid_hits,m.known_fid_total,null,m.macro_known_fid_coverage,m.returned_pairs,m.uncertain_pairs]);
 }
}
overview.getRange('A4:L12').values=summary;overview.getRange('A4:L12').format={font:{name:'Arial',size:10},rowHeight:34,verticalAlignment:'center'};
overview.getRange('A4:L4').format={fill:'#243B57',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:46,wrapText:true};
for(let r=5;r<=12;r++){
 overview.getRange(`F${r}`).formulas=[[`=IF(D${r}>0,E${r}/D${r},"")`]];
 overview.getRange(`I${r}`).formulas=[[`=IF(H${r}>0,G${r}/H${r},"")`]];
}
for(const range of ['F5:F12','I5:J12'])overview.getRange(range).setNumberFormat('0.0%');
[19,26,12,13,13,18,18,18,22,22,18,18].forEach((width,i)=>overview.getRangeByIndexes(0,i,26,1).format.columnWidth=width);
const notes=[
 '用例命中：至少召回一个预期fid。已知fid覆盖率不是完整语义召回率；只有正例标签不能计算真实精确率。',
 `参考日期：${data.scope.reference_date}。范围：时间${data.scope.categories.time}、地点${data.scope.categories.place}、组合${data.scope.categories.both}，均来自前100条之后；不是连续101–200条。`,
 '两组视觉后端均为Qwen。V2增加全库描述/OCR、结构化年份与地点检索，以及多源证据复核；不与历史DeepSeek不同查询集直接计算提升。',
 'V2保留基线正关联，召回非下降是并集性质。纯时间/地点条件可由明确结构化字段确认；视觉条件仍需图像复核。',
 `服务商图像拦截影响${provenance.provider_blocks.blocked_image_ids.length}张图片，未取得的视觉结论记待核实，不算负例。详见正式报告与溯源。`,
 `格式校正后仍无有效结论的最终待核实关系：V1 ${provenance.schema_abstentions.final_uncertain_pairs.baseline}条、V2 ${provenance.schema_abstentions.final_uncertain_pairs.hybrid}条；均未计入正式返回和命中。`,
 '矩阵：1=确认相关；0=筛选或复核未命中；?=待核实；S=目录未搜索；E=解码失败。0不是完整人工负例。',
 '数据来源：data/benchmark.json、provenance.json。完整定义、失败案例与用量见docs/evaluation-time-place.md。',
 '隐私：仅公开稳定图片ID与匿名fid映射，不含原图、原始fid、文件名、描述、GPS、私有服务端点或密钥。',
 '正关联表“已列于标签”仅标识是否在D列；未列于D列不等于语义不相关。'
];
notes.forEach((text,i)=>{const r=15+i;overview.mergeCells(`A${r}:L${r}`);overview.getRange(`A${r}`).values=[[text]];overview.getRange(`A${r}:L${r}`).format={font:{name:'Arial',size:10,color:'#475569'},wrapText:true,rowHeight:42,verticalAlignment:'center'};});
const caseRows=data.cases.map(r=>[r.id,r.query,category[r.category],r.gt_count,+r.baseline.hit_any,+r.hybrid.hit_any,r.baseline.hit_fids,r.hybrid.hit_fids,r.baseline.known_fid_recall_all,r.hybrid.known_fid_recall_all,r.baseline_image_ids.length,r.hybrid_image_ids.length,r.new_positive_image_ids.length,r.baseline_uncertain_image_ids.length,r.hybrid_uncertain_image_ids.length]);
table(cases,[['用例','查询','类别','预期fid数','V1命中','V2命中','V1已知fid命中','V2已知fid命中','V1覆盖率','V2覆盖率','V1返回','V2返回','新增返回','V1待核实','V2待核实'],...caseRows],[14,40,20,16,15,15,22,22,18,18,17,17,17,18,18],42);
cases.getRangeByIndexes(1,1,data.cases.length,1).format.wrapText=true;cases.getRangeByIndexes(1,8,data.cases.length,2).setNumberFormat('0.0%');
cases.getRangeByIndexes(1,4,data.cases.length,2).formulas=data.cases.map((_,i)=>[`=IF(G${i+2}>0,1,0)`,`=IF(H${i+2}>0,1,0)`]);
cases.getRangeByIndexes(1,8,data.cases.length,2).formulas=data.cases.map((_,i)=>[`=IF(D${i+2}>0,G${i+2}/D${i+2},"")`,`=IF(D${i+2}>0,H${i+2}/D${i+2},"")`]);
// Headline counts and rates trace directly to the case-level inputs.
const last=data.cases.length+1;const detail=column=>`'逐条结果'!${column}2:${column}${last}`;
for(let index=0;index<8;index++){
 const r=index+5;const isHybrid=index%2===1;const group=index<2?null:['时间','地点','时间与地点'][Math.floor(index/2)-1];
 const sum=column=>group?`SUMIF(${detail('C')},"${group}",${detail(column)})`:`SUM(${detail(column)})`;
 overview.getRange(`C${r}`).formulas=[[group?`=COUNTIF(${detail('C')},"${group}")`:`=COUNTA(${detail('A')})`]];
 overview.getRange(`D${r}`).formulas=[[group?`=COUNTIFS(${detail('C')},"${group}",${detail('D')},">0")`:`=COUNTIF(${detail('D')},">0")`]];
 for(const [output,input] of [['E',isHybrid?'F':'E'],['G',isHybrid?'H':'G'],['H','D'],['K',isHybrid?'L':'K'],['L',isHybrid?'O':'N']])overview.getRange(`${output}${r}`).formulas=[['='+sum(input)]];
 overview.getRange(`J${r}`).formulas=[[`=IF(D${r}>0,${sum(isHybrid?'J':'I')}/D${r},"")`]];
}
const failed=data.cases.filter(r=>r.gt_count&&!r.hybrid.hit_any);
table(failures,[['用例','查询','类别','预期fid数','V1命中fid','V2命中fid','V2返回数','V2待核实数'],...failed.map(r=>[r.id,r.query,category[r.category],r.gt_count,r.baseline.hit_fids,r.hybrid.hit_fids,r.hybrid_image_ids.length,r.hybrid_uncertain_image_ids.length])],[14,40,20,18,20,20,20,22],42);
if(failed.length)failures.getRangeByIndexes(1,1,failed.length,1).format.wrapText=true;
const associations=[];const sourceNames={visual_baseline:'原视觉分支',structured_metadata:'结构化时间地点',qwen_multimodal:'Qwen多模态复核'};
for(const r of data.cases){const baseline=new Set(r.baseline_image_ids);const known=new Set(r.ground_truth_fids.flatMap(fid=>data.fid_to_image_ids[fid]||[]));for(const image of r.hybrid_image_ids)associations.push([r.id,image,sourceNames[r.hybrid_confirmation_sources[image]]||r.hybrid_confirmation_sources[image],baseline.has(image)?0:1,known.has(image)?1:0]);}
table(pairs,[['用例','稳定图片ID','确认来源','V2新增','已列于标签'],...associations],[18,22,28,18,20]);
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'汇总!A4:L12',include:'values,formulas',tableMaxRows:9,tableMaxCols:12,maxChars:4500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'final formula error scan'})).ndjson);
if(process.env.REPORT_PREVIEW_DIR){await fs.mkdir(process.env.REPORT_PREVIEW_DIR,{recursive:true});for(const [sheetName,range,file] of [['汇总','A1:L24','summary'],['逐条结果','A1:O12','cases'],['未命中','A1:H12','failures'],['正关联','A1:E10','pairs']]){const blob=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(process.env.REPORT_PREVIEW_DIR,file+'.png'),new Uint8Array(await blob.arrayBuffer()));}}
const output=path.join(root,'outputs',process.env.CODEX_THREAD_ID||process.env.REPORT_OUTPUT_ID||crypto.randomUUID());await fs.mkdir(output,{recursive:true});const file=path.join(output,'comparison-time-place.xlsx');await(await SpreadsheetFile.exportXlsx(wb)).save(file);await fs.copyFile(file,path.join(directory,'comparison.xlsx'));
const manifest=JSON.parse(await fs.readFile(path.join(directory,'checksums.json'),'utf8'));manifest.files['comparison.xlsx']=crypto.createHash('sha256').update(await fs.readFile(file)).digest('hex');await fs.writeFile(path.join(directory,'checksums.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({output_file:file,cases:data.cases.length,positive_associations:associations.length}));
