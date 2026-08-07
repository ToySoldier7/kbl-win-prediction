import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = path.resolve(import.meta.dirname, "..");
const processedDir = path.join(projectRoot, "data", "processed");
const payload = JSON.parse(await fs.readFile(path.join(processedDir, "workbook_payload.json"), "utf8"));

const workbook = Workbook.create();
workbook.comments.setSelf({ displayName: "User" });
const summary = workbook.worksheets.add("요약");
const games = workbook.worksheets.add("경기");
const teamGames = workbook.worksheets.add("팀경기");
const pregame = workbook.worksheets.add("경기전피처");
const sequenceCarry = workbook.worksheets.add("시퀀스_연결");
const sequenceReset = workbook.worksheets.add("시퀀스_초기화");
const elo = workbook.worksheets.add("Elo");

const navy = "#16324F";
const blue = "#2F75B5";
const lightBlue = "#DCE6F1";
const green = "#E2F0D9";
const red = "#FCE4D6";
const gray = "#F2F2F2";
const white = "#FFFFFF";
const border = "#C9D2DC";

function writeMatrix(sheet, matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  sheet.getRangeByIndexes(0, 0, rows, cols).values = matrix;
  return { rows, cols };
}

function styleDataSheet(sheet, matrix, tableName, percentHeaders = []) {
  const { rows, cols } = writeMatrix(sheet, matrix);
  const used = sheet.getRangeByIndexes(0, 0, rows, cols);
  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format = {
    fill: navy,
    font: { bold: true, color: white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: border },
  };
  header.format.rowHeightPx = 34;
  used.format.font = { name: "Aptos", size: 9 };
  used.format.autofitColumns();
  for (let col = 0; col < cols; col += 1) {
    const column = sheet.getRangeByIndexes(0, col, rows, 1);
    column.format.columnWidthPx = Math.min(Math.max(column.format.columnWidthPx || 80, 70), 155);
  }
  const headers = matrix[0];
  for (const name of percentHeaders) {
    const col = headers.indexOf(name);
    if (col >= 0 && rows > 1) sheet.getRangeByIndexes(1, col, rows - 1, 1).format.numberFormat = "0.0%";
  }
  const dateCol = headers.findIndex((name) => name === "game_date" || name === "target_date");
  if (dateCol >= 0 && rows > 1) sheet.getRangeByIndexes(1, dateCol, rows - 1, 1).format.numberFormat = "yyyy-mm-dd";
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  const lastCell = sheet.getRangeByIndexes(rows - 1, cols - 1, 1, 1).address.split("!").pop();
  sheet.tables.add(`A1:${lastCell}`, true, tableName).style = "TableStyleMedium2";
}

styleDataSheet(games, payload.games, "GamesTable");
styleDataSheet(
  teamGames,
  payload.team_games,
  "TeamGamesTable",
  [
    "fg_pct",
    "three_pt_pct",
    "free_throw_pct",
    "efg_pct",
    "tov_pct",
    "oreb_pct",
    "fta_rate",
    "rolling_win_5",
    "elo_win_probability",
  ],
);
teamGames.getRangeByIndexes(1, 3, payload.team_games.length - 1, 1).format.numberFormat = "00";
styleDataSheet(
  pregame,
  payload.pregame,
  "PregameTable",
  [
    "home_rolling_win_5",
    "away_rolling_win_5",
    "diff_rolling_efg_pct_5",
    "diff_rolling_tov_pct_5",
    "diff_rolling_oreb_pct_5",
    "diff_rolling_fta_rate_5",
    "elo_home_win_probability",
  ],
);
styleDataSheet(sequenceCarry, payload.sequences_carryover, "SequenceCarryTable");
styleDataSheet(sequenceReset, payload.sequences_reset, "SequenceResetTable");
styleDataSheet(elo, payload.elo_predictions, "EloTable", ["elo_home_win_probability"]);

summary.showGridLines = false;
summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["KBL LSTM/GRU 데이터 품질 보고서"]];
summary.getRange("A1:F1").format = {
  fill: navy,
  font: { bold: true, color: white, size: 18 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("A1:F1").format.rowHeightPx = 42;

summary.getRange("A3:B3").values = [["핵심 지표", "값"]];
summary.getRange("A4:A10").values = [
  ["전체 경기"],
  ["팀 관점 행"],
  ["연장전 경기"],
  ["경기 전 모델 행"],
  ["N=5 연결형 시퀀스"],
  ["N=5 초기화형 시퀀스"],
  ["전체 검사"],
];
summary.getRange("B4:B9").values = [[
  payload.report.total_games,
], [
  payload.report.team_game_rows,
], [
  payload.report.overtime_games,
], [
  payload.report.pregame_model_rows,
], [
  payload.report.sequence_samples.carryover["5"],
], [
  payload.report.sequence_samples.season_reset["5"],
]];
summary.getRange("B10").values = [[payload.report.all_checks_passed ? "PASS" : "CHECK"]];
summary.getRange("A3:B3").format = { fill: blue, font: { bold: true, color: white } };
summary.getRange("A4:A10").format = { fill: lightBlue, font: { bold: true } };
summary.getRange("B4:B10").format = { fill: green, font: { bold: true, size: 12 }, horizontalAlignment: "center" };

summary.getRange("A13:D13").values = [["시즌", "실제 경기", "기대 경기", "결과"]];
const seasonNames = Object.keys(payload.report.seasons);
const seasonStartRow = 14;
const seasonEndRow = seasonStartRow + seasonNames.length - 1;
for (let i = 0; i < seasonNames.length; i += 1) {
  const row = seasonStartRow + i;
  summary.getRange(`A${row}`).values = [[seasonNames[i]]];
  summary.getRange(`B${row}`).values = [[payload.report.seasons[seasonNames[i]]]];
  summary.getRange(`C${row}`).values = [[payload.report.expected_seasons[seasonNames[i]]]];
  summary.getRange(`D${row}`).formulas = [[`=IF(B${row}=C${row},"PASS","CHECK")`]];
}
summary.getRange("A13:D13").format = { fill: blue, font: { bold: true, color: white } };
summary.getRange(`D${seasonStartRow}:D${seasonEndRow}`).format = { fill: green, font: { bold: true } };

const checkEntries = Object.entries(payload.report.checks);
const checksHeaderRow = seasonEndRow + 3;
const checksStartRow = checksHeaderRow + 1;
const checksEndRow = checksStartRow + checkEntries.length - 1;
summary.getRange(`A${checksHeaderRow}:B${checksHeaderRow}`).values = [["품질 검사", "결과"]];
summary.getRange(`A${checksHeaderRow}:B${checksHeaderRow}`).format = { fill: blue, font: { bold: true, color: white } };
for (let i = 0; i < checkEntries.length; i += 1) {
  const row = checksStartRow + i;
  summary.getRange(`A${row}:B${row}`).values = [[checkEntries[i][0], checkEntries[i][1] ? "PASS" : "FAIL"]];
  summary.getRange(`B${row}`).format = {
    fill: checkEntries[i][1] ? green : red,
    font: { bold: true },
    horizontalAlignment: "center",
  };
}

summary.getRange("D3:F3").merge();
summary.getRange("D3").values = [["데이터 출처 및 설계"]];
summary.getRange("D3:F3").format = { fill: blue, font: { bold: true, color: white }, horizontalAlignment: "center" };
summary.getRange("D4:F8").merge();
summary.getRange("D4").values = [[
  "2013-14~2024-25 정규리그를 경기 단위로 수집했습니다. Four Factors·연승/연패·이동평균·Elo와 구단 코드 변경을 연결하는 franchise_id를 제공하며, 시즌 연결형과 초기화형 시퀀스를 모두 제공합니다.",
]];
summary.getRange("D4:F8").format = { fill: gray, wrapText: true, verticalAlignment: "top" };
const sourceHeaderRow = checksHeaderRow;
const eloHeaderRow = checksHeaderRow + 4;
summary.getRange(`D${eloHeaderRow}:F${eloHeaderRow}`).merge();
summary.getRange(`D${eloHeaderRow}`).values = [["Elo 테스트 기준선"]];
summary.getRange(`D${eloHeaderRow}:F${eloHeaderRow}`).format = { fill: blue, font: { bold: true, color: white }, horizontalAlignment: "center" };
summary.getRange(`D${eloHeaderRow + 1}:E${eloHeaderRow + 4}`).values = [
  ["정확도", payload.report.elo_baseline.test.accuracy],
  ["Brier score", payload.report.elo_baseline.test.brier_score],
  ["Log loss", payload.report.elo_baseline.test.log_loss],
  ["테스트 샘플", payload.report.elo_baseline.test.samples],
];
summary.getRange(`D${eloHeaderRow + 1}:D${eloHeaderRow + 4}`).format = { fill: lightBlue, font: { bold: true } };
summary.getRange(`E${eloHeaderRow + 1}:E${eloHeaderRow + 3}`).format.numberFormat = "0.0000";
summary.getRange(`E${eloHeaderRow + 1}:E${eloHeaderRow + 4}`).format = { fill: green, font: { bold: true } };

summary.getRange(`D${sourceHeaderRow}:F${sourceHeaderRow}`).merge();
summary.getRange(`D${sourceHeaderRow}`).values = [["공식 출처"]];
summary.getRange(`D${sourceHeaderRow}:F${sourceHeaderRow}`).format = { fill: blue, font: { bold: true, color: white }, horizontalAlignment: "center" };
summary.getRange(`D${sourceHeaderRow + 1}:F${sourceHeaderRow + 1}`).merge();
summary.getRange(`D${sourceHeaderRow + 1}`).values = [["https://www.kbl.or.kr/match/schedule"]];
summary.getRange(`D${sourceHeaderRow + 2}:F${sourceHeaderRow + 2}`).merge();
summary.getRange(`D${sourceHeaderRow + 2}`).values = [["https://kbl.or.kr/rule/operation"]];
summary.getRange(`D${sourceHeaderRow + 1}:F${sourceHeaderRow + 2}`).format = { fill: gray, font: { color: "#0563C1" } };
workbook.comments.addThread({ cell: summary.getRange(`D${sourceHeaderRow + 1}`) }, "Source: KBL 공식 일정 및 결과 페이지");

const summaryEndRow = Math.max(checksEndRow, eloHeaderRow + 4);
summary.getRange(`A3:F${summaryEndRow}`).format.borders = { preset: "outside", style: "thin", color: border };
summary.getRange(`A1:F${summaryEndRow}`).format.font = { name: "Aptos" };
summary.getRange("A:A").format.columnWidthPx = 340;
summary.getRange("B:B").format.columnWidthPx = 105;
summary.getRange("C:C").format.columnWidthPx = 95;
summary.getRange("D:F").format.columnWidthPx = 125;
summary.freezePanes.freezeRows(1);

const inspect = await workbook.inspect({
  kind: "table",
  sheetId: "요약",
  range: `A1:F${summaryEndRow}`,
  include: "values,formulas",
  tableMaxRows: summaryEndRow,
  tableMaxCols: 6,
  maxChars: 8000,
});
console.log(inspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const previews = [
  ["요약", `A1:F${summaryEndRow}`],
  ["경기", "A1:N22"],
  ["팀경기", "A1:AF22"],
  ["경기전피처", "A1:S22"],
  ["시퀀스_연결", "A1:K22"],
  ["시퀀스_초기화", "A1:K22"],
  ["Elo", "A1:I22"],
];
for (const [sheetName, range] of previews) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(processedDir, `preview_${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(processedDir, "kbl_data_quality.xlsx"));
console.log("Workbook created");
