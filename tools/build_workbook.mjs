// Optional workbook builder: requires the Codex bundled @oai/artifact-tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base=path.join(root,'benchmarks/2026-11-17');
const data=JSON.parse(await fs.readFile(path.join(base,'data/benchmark.json'),'utf8'));
const wb=Workbook.create();
const summary=wb.worksheets.add('汇总');
const cases=wb.worksheets.add('逐条结果');
const misses=wb.worksheets.add('未命中分析');
const matches=wb.worksheets.add('正关联');
function table(sheet,rows,widths,height=34){
 sheet.showGridLines=false;
 const all=sheet.getRangeByIndexes(0,0,rows.length,rows[0].length);all.values=rows;
 all.format={font:{name:'Arial',size:10},rowHeight:height,verticalAlignment:'center'};
 sheet.getRangeByIndexes(0,0,1,rows[0].length).format={fill:'#243B57',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:44,wrapText:true};
 widths.forEach((width,i)=>sheet.getRangeByIndexes(0,i,rows.length,1).format.columnWidth=width);
 sheet.freezePanes.freezeRows(1);
}
summary.showGridLines=false;summary.mergeCells('A2:I2');
summary.getRange('A2').values=[['本地图片检索工作流测试对比  2026-11-17']];summary.getRange('A2').format.font={name:'Arial',size:15,bold:true};
const overview=[['范围','方法','有标签用例数','命中用例数','用例命中率','返回关联数','已知fid命中','已知fid总数','已知fid覆盖率']];
for(const [scope,key] of [['新增11–100','holdout_011_100'],['累计1–100','combined_001_100'],['开发1–10','development_001_010']]){
 for(const [method,label] of [['skill','DeepSeek Skill'],['codex','Codex辅助审图样本']]){
  const m=data.summaries[key][method];overview.push([scope,label,m.evaluated_cases,m.hit_cases,null,m.returned_pairs,m.known_fid_hits,m.known_fid_total,null]);
 }
}
overview.push(['开发1–10','元数据规则基线',10,10,null,970,453,842,null]);
summary.getRange('A4:I11').values=overview;
summary.getRange('A4:I11').format={font:{name:'Arial',size:10},rowHeight:34,verticalAlignment:'center'};
summary.getRange('A4:I4').format={fill:'#243B57',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:44,wrapText:true};
for(let row=5;row<=11;row++){
 summary.getRange(`E${row}`).formulas=[[`=D${row}/C${row}`]];
 summary.getRange(`I${row}`).formulas=[[`=G${row}/H${row}`]];
}
summary.getRange('E5:E11').setNumberFormat('0.0%');summary.getRange('I5:I11').setNumberFormat('0.0%');
summary.getRange('A1:A23').format.columnWidth=22;summary.getRange('B1:B23').format.columnWidth=28;summary.getRange('C1:I23').format.columnWidth=16;
const notes=[
 '指标定义：至少一个预期fid映射图片进入正式返回集合即记命中；不等于真实精确率或全部相关图片的完整召回。',
 'Codex新增90条实际检查334次用例—图片组合、267张不同图片，候选使用ES已有描述；Skill不读取已有图片语义描述。',
 '前10条用于开发，新增查询含近义复述。两种方法输入及候选范围不同，结果不构成相同输入条件下的模型能力排名。',
 'Skill矩阵：1=相关，0=筛选或复核未命中，?=待核实，S=目录未搜索，E=解码失败。Codex矩阵只记正关联，空白不是负例。',
 '原始标签含疑似时间、颜色和物种冲突。计分后诊断未改变标注、预测或正式成绩，详见未命中分析。',
 '发布版入口增加导出、缓存和阶段校验等工程修复；本表分数来自历史冻结实验，未对发布版重新执行真实模型评测。',
 '数据来源与复算：benchmarks/2026-11-17/data/benchmark.json；完整方法与限制：docs/evaluation-report.md。',
 '隐私处理：只公开稳定图片ID与匿名fid映射，不包含原图、原始fid、原始文件名、caption或私有绝对路径。',
 '正关联表的已知标签标记仅表示是否列于D列；未列于D列不能直接判定为不相关。'
];
notes.forEach((text,i)=>{const row=14+i;summary.mergeCells(`A${row}:I${row}`);summary.getRange(`A${row}`).values=[[text]];summary.getRange(`A${row}:I${row}`).format={font:{name:'Arial',size:10,color:'#475569'},rowHeight:42,wrapText:true,verticalAlignment:'center'};});
const crows=data.cases.map(r=>[r.id,r.query,r.split==='holdout'?'新增验证集':'开发集',r.gt_count,r.skill.locally_mapped_fids,r.skill_count,+r.skill.hit_any,r.skill.known_fid_recall_all,r.codex_count,+r.codex.hit_any,r.codex.known_fid_recall_all,r.uncertain_count]);
table(cases,[['用例','查询','集合','预期fid数','可映射fid数','Skill返回数','Skill命中','Skill已知fid覆盖','Codex返回数','Codex命中','Codex已知fid覆盖','待核实数'],...crows],[13,36,16,16,18,17,15,23,18,16,24,17],42);
cases.getRange('B2:B101').format.wrapText=true;cases.getRange('H2:H101').setNumberFormat('0.0%');cases.getRange('K2:K101').setNumberFormat('0.0%');
const explanations={
 TC0009:'已知正例在拼图阶段未进入候选；开发集漏召回。',
 TC0023:'Codex有限样本未覆盖已标注fid；311个预期正例只返回1张。',
 TC0040:'Codex将两个主题按同图条件理解；组合查询需明确并集与同图语义。',
 TC0050:'27个预期图中仅两张时间字段为2010，画面为老虎；其他图时间均非2010。原始成绩保留，需复核标签与时间字段。',
 TC0051:'I02671在拼图筛选阶段漏掉；预期集合还包含蛙类和鳄鱼画面，物种标签需复核。',
 TC0054:'Codex有限样本未覆盖已标注fid；281个预期正例只返回1张。',
 TC0078:'Codex返回4张均未命中已标注fid，26个预期fid中7个无法映射本地。',
 TC0100:'预期集合包含雪地中的棕褐色麻雀及景观、线稿；需核对颜色条件与标签，原始成绩保留。'
};
const failed=data.cases.filter(r=>!r.skill.hit_any||!r.codex.hit_any);
table(misses,[['用例','查询','Skill命中','Codex命中','Skill返回','Codex返回','待核实','计分后复查说明'],...failed.map(r=>[r.id,r.query,+r.skill.hit_any,+r.codex.hit_any,r.skill_count,r.codex_count,r.uncertain_count,explanations[r.id]])],[13,36,16,17,17,18,17,100],60);
misses.getRangeByIndexes(1,7,failed.length,1).format.wrapText=true;
const positives=[];
for(const r of data.cases){
 const known=new Set(r.ground_truth_fids.flatMap(fid=>data.fid_to_image_ids[fid]||[]));
 for(const [method,label] of [['skill','DeepSeek Skill'],['codex','Codex辅助审图']])for(const id of r[method+'_image_ids'])positives.push([label,r.id,id,known.has(id)?1:0]);
}
table(matches,[['方法','用例','稳定图片ID','已列于预期标签'],...positives],[27,17,20,24]);
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'汇总!A4:I11',include:'values,formulas',tableMaxRows:8,tableMaxCols:9,maxChars:3600})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'formula error scan'})).ndjson);
if(process.env.REPORT_PREVIEW_DIR){
 await fs.mkdir(process.env.REPORT_PREVIEW_DIR,{recursive:true});
 for(const [sheetName,range,name] of [['汇总','A1:I22','summary'],['逐条结果','A1:L12','cases'],['未命中分析','A1:H9','misses'],['正关联','A1:D9','matches']]){
  const blob=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(process.env.REPORT_PREVIEW_DIR,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
 }
}
const output=path.join(root,'outputs',process.env.CODEX_THREAD_ID||'public-report');await fs.mkdir(output,{recursive:true});
const file=path.join(output,'comparison-100.xlsx');await (await SpreadsheetFile.exportXlsx(wb)).save(file);
await fs.copyFile(file,path.join(base,'comparison-100.xlsx'));
const manifest=JSON.parse(await fs.readFile(path.join(base,'checksums.json'),'utf8'));
manifest.files['comparison-100.xlsx']=crypto.createHash('sha256').update(await fs.readFile(file)).digest('hex');manifest.scope='All published benchmark artifacts except this checksum manifest.';
await fs.writeFile(path.join(base,'checksums.json'),JSON.stringify(manifest,null,2)+'\n');
console.log('Exported public workbook with '+positives.length+' anonymized positive associations.');
