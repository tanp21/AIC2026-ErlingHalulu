import "@testing-library/jest-dom/vitest";

Object.defineProperty(HTMLMediaElement.prototype, "pause", { configurable: true, value() {} });
Object.defineProperty(HTMLMediaElement.prototype, "load", { configurable: true, value() {} });

Object.defineProperty(Element.prototype, "scrollIntoView", { configurable: true, value() {} });

// jsdom ships no object URLs; blob downloads and image previews need them.
Object.defineProperty(URL, "createObjectURL", { configurable: true, value: () => "blob:stub" });
Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value() {} });
