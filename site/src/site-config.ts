import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'understanding-dataset-lab',
  description:
    'A prototype lab for cognitive-empathy test scenarios: multi-party scenes planned by code, with answer keys that come from the plan, never from a model.',
  logoBadge: 'UD',
  brandName: 'understanding-dataset-lab',
  repoUrl: 'https://github.com/mcp-tool-shop-org/understanding-dataset-lab',
  footerText:
    'MIT Licensed — built by <a href="https://github.com/mcp-tool-shop-org" style="color:var(--color-muted);text-decoration:underline">mcp-tool-shop-org</a>. Definition of empathy from <a href="https://en.wiktionary.org/wiki/empathy" style="color:var(--color-muted);text-decoration:underline">Wiktionary</a>, CC BY-SA 4.0.',

  hero: {
    badge: 'Prototype · Python 3.15+ · MIT',
    headline: 'Who knows what,',
    headlineAccent: 'proved by the plan.',
    description:
      'udlab builds short scenes with several people, planned by code and put into prose by a local writer model. The key to every question, who saw what and who believes what, comes from the code\'s plan and is sealed with a private seed. A prototype: no sealed result and no trained model yet.',
    primaryCta: { href: 'https://github.com/mcp-tool-shop-org/understanding-dataset-lab', label: 'View on GitHub' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      {
        label: 'Generate',
        code: 'uv run udlab-gen --count 3 --out-dir artifacts/demo\n\n# plans 3 scenes, writes them with the CPU mock,\n# and seals their keys. No GPU needed.',
      },
      {
        label: 'Probe',
        code: 'uv run udlab-gen --probe-only \\\n  artifacts/demo/sealed_store/keys.sealed.jsonl\n\n# tests whether surface features\n# give the answers away.',
      },
      {
        label: 'Test',
        code: 'uv run pytest\n\n# 759 tests, synthetic seeds only.',
      },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'principles',
      title: 'How it works',
      subtitle: 'The answer is computed from the plan, not judged.',
      features: [
        {
          title: 'Code-made keys',
          desc: 'Who moved where, who could see and who heard what are planned events, so what each person believes follows mechanically. No model writes or grades a key.',
        },
        {
          title: 'Checked prose',
          desc: 'A local writer model puts each scene into words. Every rendering is checked by code against the plan before it counts.',
        },
        {
          title: 'Sealed and probed',
          desc: 'Keys are sealed with a private seed, and out-of-fold probes test whether surface features give the answers away.',
        },
      ],
    },
    {
      kind: 'features',
      id: 'status',
      title: 'Status: a prototype',
      subtitle: 'Counts on development banks of 160 renderings, negative results first.',
      features: [
        {
          title: 'Stage 3: not adopted',
          desc: 'Listing accepted sentence forms in the prompt: 111 of 160 passed, against 127 for the baseline.',
        },
        {
          title: 'Stage 4: not adopted',
          desc: 'Sentence shapes assigned by code: 89 of 160, against 120. The development gate failed.',
        },
        {
          title: 'Stage 5: passed its gates',
          desc: 'Code moves and rewritten speech passed all 7 pre-registered development gates. No sealed confirmatory run yet.',
        },
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'Try it without a GPU',
      cards: [
        {
          title: 'Clone and run',
          code: 'git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab\ncd understanding-dataset-lab\nuv run udlab-gen --count 3 --out-dir artifacts/demo',
        },
        {
          title: 'Options',
          code: 'uv run udlab-gen --help',
        },
      ],
    },
  ],
};
