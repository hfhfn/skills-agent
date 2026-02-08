import { memo } from "react";

import type { SkillSummary } from "../state/chatReducer";

type SkillPanelProps = {
  skills: SkillSummary[];
  activeSkillName?: string;
  loading: boolean;
  error?: string;
};

export const SkillPanel = memo(function SkillPanel({
  skills,
  activeSkillName,
  loading,
  error,
}: SkillPanelProps) {
  return (
    <aside className="skills-panel">
      <header className="skills-panel__header">
        <p className="eyebrow">能力</p>
        <h2>已发现技能</h2>
      </header>

      {loading && <p className="skills-panel__hint">正在扫描技能...</p>}
      {error && <p className="skills-panel__error">{error}</p>}

      {!loading && !error && skills.length === 0 && (
        <p className="skills-panel__hint">未发现可用技能。</p>
      )}

      <ul className="skills-list">
        {skills.map((skill) => {
          const active = skill.name === activeSkillName;
          const description = skill.description || "暂无描述";
          return (
            <li
              key={skill.name}
              className={active ? "skill-card skill-card--active" : "skill-card"}
            >
              <div className="skill-card__title-row">
                <h3>{skill.name}</h3>
                {active && <span className="skill-card__badge">使用中</span>}
              </div>
              <p className="skill-card__description" title={description}>
                {description}
              </p>
              <code className="skill-card__path" title={skill.path}>
                {skill.path}
              </code>
            </li>
          );
        })}
      </ul>
    </aside>
  );
});
