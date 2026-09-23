import { describe, expect, it } from "vitest";

import { fold, highlight } from "./highlight";

const lit = (text: string, keywords: string[]) =>
  highlight(text, keywords).segments.filter((segment) => segment.hit).map((segment) => [segment.text, segment.hit]);

describe("fold", () => {
  it("drops accents and case without moving any offset", () => {
    expect(fold("Đèo Tà Pứa")).toBe("deo ta pua");
    expect(fold("Đèo Tà Pứa")).toHaveLength("Đèo Tà Pứa".length);
    // a decomposed input (e + combining circumflex) folds to the same length as its NFC form
    expect(fold("Lê")).toBe("le");
  });
});

describe("highlight", () => {
  it("lights a whole phrase, accents ignored on either side", () => {
    expect(lit("hôm nay nấu cá kho tộ", ["cá kho"])).toEqual([["cá kho", "phrase"]]);
    expect(lit("cho thêm hai muỗng nước tương", ["nuoc tuong"])).toEqual([["nước tương", "phrase"]]);
  });

  it("lights a lone informative word of a keyword, but never a stopword", () => {
    const result = highlight("thuyền của ông ở trên sông", ["thuyền màu đỏ trên sông"]);
    expect(result.segments.filter((s) => s.hit).map((s) => s.text)).toEqual(["thuyền", "sông"]);
    expect(result.segments.some((s) => s.hit && s.text.trim() === "trên")).toBe(false);
  });

  it("treats a number as informative even when it is short", () => {
    expect(lit("giá 15.000 đồng", ["15.000"])).toEqual([["15.000", "phrase"]]);
    expect(lit("có 3 con", ["3 con thuyền"]).map(([text]) => text)).toEqual(["3"]);  // "con" is a function word
  });

  it("keeps the original text exactly, however it is split", () => {
    const text = "Thuyền đỏ, trên sông Hương; 130.000 đồng.";
    const { segments } = highlight(text, ["thuyền đỏ", "sông Hương", "130.000"]);
    expect(segments.map((segment) => segment.text).join("")).toBe(text);
  });

  it("reports what matched, the keyword as written for a phrase and the word for a lone hit", () => {
    const { matched } = highlight("nấu cá kho với hành", ["cá kho", "hành tím"]);
    expect(matched).toEqual(["cá kho", "hành"]);
  });

  it("returns the text unmarked when nothing matches or there are no keywords", () => {
    expect(highlight("xin chào", ["cá kho"]).segments).toEqual([{ text: "xin chào", hit: null }]);
    expect(highlight("xin chào", []).matched).toEqual([]);
    expect(highlight("", ["a b"]).segments).toEqual([{ text: "", hit: null }]);
  });

  it("matches English caption words case-insensitively", () => {
    expect(lit("A Red boat sails down the River", ["a red boat on a river"]).map(([text]) => text))
      .toEqual(["Red boat", "River"]);
  });
});
