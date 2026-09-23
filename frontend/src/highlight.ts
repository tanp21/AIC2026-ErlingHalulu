/**
 * Light up, in a transcript line, an OCR string or a caption, the words a query is about.
 *
 * The search side folds accents and case before it compares (`vi_folded` in Elasticsearch),
 * so a reviewer who sees "nuoc tuong" match "nước tương" in the ranking should see it lit up
 * here too. Folding is done one character at a time so every offset in the folded text is
 * an offset in the original.
 */

export interface Segment { text: string; hit: "phrase" | "word" | null }
export interface Highlighted { segments: Segment[]; matched: string[] }

// Words that say nothing about a video. Folded, both languages: transcripts are Vietnamese,
// captions English, and the plan's keywords are a mix of query words and quoted strings.
const STOPWORDS = new Set([
  "va", "la", "cua", "co", "o", "trong", "cac", "mot", "nhung", "cho", "voi", "duoc", "den", "tu", "khi",
  "nay", "do", "ma", "thi", "rat", "da", "se", "dang", "cung", "nhu", "de", "the", "nao", "gi", "bao",
  "nhieu", "ve", "tren", "duoi", "ra", "vao", "len", "xuong", "hay", "hoac", "nhung", "neu", "vi", "nen",
  "con", "nguoi", "mon", "kia", "ai", "ban", "toi", "minh", "roi", "khong", "lam", "cai", "chiec", "xem",
  "hom", "vay", "nhe", "a", "oi", "day", "kia", "sao", "ma", "cung", "van", "moi", "cu", "luon",
  "a", "an", "of", "in", "on", "at", "and", "is", "are", "to", "with", "his", "her", "its", "for", "by",
  "from", "that", "this", "it", "as", "be", "has", "have", "or", "there", "their", "they", "he", "she",
]);

function foldChar(char: string): string {
  if (char === "đ" || char === "Đ") return "d";
  const base = char.normalize("NFD").replace(/[̀-ͯ]/g, "");
  return (base[0] ?? char).toLowerCase();
}

/** Lower-case, accent-free, same length as the NFC form of `text`. */
export function fold(text: string): string {
  let folded = "";
  for (const char of text.normalize("NFC")) folded += char.length === 1 ? foldChar(char) : foldChar(char[0]).padEnd(char.length, " ");
  return folded;
}

interface Token { value: string; start: number; end: number }

function tokens(folded: string): Token[] {
  return Array.from(folded.matchAll(/[a-z0-9]+/g), (match) => ({
    value: match[0], start: match.index ?? 0, end: (match.index ?? 0) + match[0].length,
  }));
}

function informative(token: string): boolean {
  return !STOPWORDS.has(token) && (token.length >= 2 || /\d/.test(token));
}

/**
 * Segments of `text` with `keywords` marked. A whole keyword phrase found in order is a
 * "phrase" hit; a single informative word of any keyword is a "word" hit. `matched` lists
 * the keyword phrases found whole, then the lone words, as the keywords wrote them.
 */
export function highlight(text: string, keywords: string[]): Highlighted {
  const source = text.normalize("NFC");
  const folded = fold(source);
  const found = tokens(folded);
  const marks: Array<"phrase" | "word" | null> = new Array(found.length).fill(null);
  // Which phrase occurrence a token belongs to, so "15.000" stays one lit run rather than
  // "15" and "000" split by the point between them.
  const runs: number[] = new Array(found.length).fill(-1);
  let run = 0;
  const matched: string[] = [];
  const words = new Map<string, string>();

  for (const keyword of keywords) {
    const parts = tokens(fold(keyword));
    for (const part of parts) if (informative(part.value) && !words.has(part.value)) words.set(part.value, keyword);
    if (parts.length < 2) continue;
    for (let at = 0; at + parts.length <= found.length; at += 1) {
      if (parts.every((part, offset) => found[at + offset].value === part.value)) {
        for (let offset = 0; offset < parts.length; offset += 1) { marks[at + offset] = "phrase"; runs[at + offset] = run; }
        run += 1;
        if (!matched.includes(keyword)) matched.push(keyword);
      }
    }
  }
  found.forEach((token, index) => {
    if (marks[index] === null && words.has(token.value)) {
      marks[index] = "word";
      const label = source.slice(token.start, token.end).toLowerCase();
      if (!matched.includes(label) && !matched.includes(words.get(token.value) ?? "")) matched.push(label);
    }
  });

  const segments: Segment[] = [];
  let cursor = 0;
  const push = (end: number, hit: Segment["hit"]) => {
    if (end > cursor) {
      const last = segments[segments.length - 1];
      if (last && last.hit === hit) last.text += source.slice(cursor, end);
      else segments.push({ text: source.slice(cursor, end), hit });
      cursor = end;
    }
  };
  found.forEach((token, index) => {
    if (marks[index] === null) return;
    if (!(index > 0 && runs[index] >= 0 && runs[index] === runs[index - 1])) push(token.start, null);
    push(token.end, marks[index]);
  });
  push(source.length, null);
  // Two marked words split only by a space read as one lit phrase.
  const merged: Segment[] = [];
  for (const segment of segments) {
    const last = merged[merged.length - 1];
    const bridge = last && last.hit && segment.hit === null && segment.text.trim() === "";
    if (bridge) { last.text += segment.text; continue; }
    if (last && last.hit && segment.hit && last.text.endsWith(" ")) { last.text += segment.text; if (segment.hit === "phrase") last.hit = "phrase"; continue; }
    merged.push({ ...segment });
  }
  return { segments: merged.length ? merged : [{ text: source, hit: null }], matched };
}
